from ...utils.make_model import PLMBaseModel, align_token_span
from ...layers.classifier import EfficientGlobalPointer, EfficientBiaffineSpanClassifier
from ...layers.loss import SparseMultiLabelCrossEntropy, MultiLabelCategoricalCrossEntropy
from ...layers.dropout import MultiDropout
from ...metrics.event import EventF1, Event, Role
from ...metrics.span import SpanF1
import torch
from itertools import groupby
from typing import Any, Union, Optional, List, Tuple
import numpy as np


class DedupList(list):
    """定义去重的list
    """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    def neighbors(host, argus, links):
        """构建邻集（host节点与其所有邻居的集合）
        """
        results = [host]
        for argu in argus:
            if host[2:] + argu[2:] in links:
                results.append(argu)
        return list(sorted(results))
    
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]

class GPLinkerForEventExtraction(PLMBaseModel):
    def __init__(self, 
                 lr: float = 3e-5,
                 hidden_size: int = 64,
                 weight_decay: float = 0.01,
                 threshold: float = 0.0,
                 scheduler: str = 'linear_warmup',
                 **kwargs: Any) -> None:
        super().__init__()
        
        self.plm = self.get_plm_architecture()

        self.role_classifier = EfficientGlobalPointer(input_size=self.plm.config.hidden_size,
                                                      hidden_size=hidden_size,
                                                      output_size=len(self.hparams.id2label))
        
        self.head_classifier = EfficientGlobalPointer(input_size=self.plm.config.hidden_size,
                                                      hidden_size=hidden_size,
                                                      output_size=1,
                                                      add_rope=False)
        
        self.tail_classifier = EfficientGlobalPointer(input_size=self.plm.config.hidden_size,
                                                      hidden_size=hidden_size,
                                                      output_size=1,
                                                      add_rope=False)
        
        
        self.dropout = MultiDropout()
        
        self.role_criterion = MultiLabelCategoricalCrossEntropy()
        self.head_criterion = MultiLabelCategoricalCrossEntropy()
        self.tail_criterion = MultiLabelCategoricalCrossEntropy()
    
        self.event_metric = EventF1()             
        self.role_metric = SpanF1()
        self.head_metric = SpanF1()
        self.tail_metric = SpanF1()

        
    def forward(self, input_ids, attention_mask):
        x = self.plm(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        role_logits = self.role_classifier(x, mask=attention_mask)
        head_logits = self.head_classifier(x, mask=attention_mask)
        tail_logits = self.tail_classifier(x, mask=attention_mask)
        return role_logits, head_logits, tail_logits


    def step(self, batch):
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        role_logits, head_logits, tail_logits = self(input_ids, attention_mask)
        
        role_true = batch['role_tags']
        head_true = batch['head_tags']
        tail_true = batch['tail_tags']
        
        b,n,s,s = role_true.shape
        role_loss = self.role_criterion(role_logits.reshape(-1, s*s), role_true.reshape(-1, s*s))
        head_loss = self.head_criterion(head_logits, head_true)
        tail_loss = self.tail_criterion(tail_logits, tail_true)
        loss = (role_loss + head_loss + tail_loss) / 3
        
        return loss, role_logits, head_logits, tail_logits, role_true, head_true, tail_true


    def training_step(self, batch, batch_idx):
        loss, role_logits, head_logits, tail_logits, role_true, head_true, tail_true = self.step(batch)
        
        role_pred = role_logits.gt(self.hparams.threshold).float()
        head_pred = head_logits.gt(self.hparams.threshold).float()
        tail_pred = tail_logits.gt(self.hparams.threshold).float()

        
        self.role_metric(role_pred, role_true)
        self.head_metric(head_pred, head_true)
        self.tail_metric(tail_pred, tail_true)
        
        self.log('train/role_f1', self.role_metric, on_step=True, prog_bar=True)
        self.log('train/head_f1', self.head_metric, on_step=True, prog_bar=True)
        self.log('train/tail_f1', self.tail_metric, on_step=True, prog_bar=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        loss, role_logits, head_logits, tail_logits, role_true, head_true, tail_true = self.step(batch)
        batch_true_events = self.extract_events(role_true, head_true, tail_true)
        batch_pred_events = self.extract_events(role_logits, head_logits, tail_logits)
        self.event_metric(batch_pred_events, batch_true_events)
        self.log('val/f1', self.event_metric, on_step=False, on_epoch=True, prog_bar=True)
        
        role_pred = role_logits.gt(self.hparams.threshold).float()
        head_pred = head_logits.gt(self.hparams.threshold).float()
        tail_pred = tail_logits.gt(self.hparams.threshold).float()
        self.role_metric(role_pred, role_true)
        self.head_metric(head_pred, head_true)
        self.tail_metric(tail_pred, tail_true)
        self.log('val/role_f1', self.role_metric, on_epoch=True, prog_bar=True)
        self.log('val/head_f1', self.head_metric, on_epoch=True, prog_bar=True)
        self.log('val/tail_f1', self.tail_metric, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}


    def extract_events(self, role_logits: torch.Tensor, head_logits: torch.Tensor, tail_logits: torch.Tensor, threshold: Union[None, float]= None, mapping: Optional[List[Tuple]]= None):
        if threshold is None:
            threshold = self.hparams.threshold
        role_logits = role_logits.cpu().detach().numpy()
        head_logits = head_logits.cpu().detach().numpy()
        tail_logits = tail_logits.cpu().detach().numpy()
        assert len(head_logits) == len(tail_logits) == len(role_logits)
        batch_events = []
        for role_logit, head_logit, tail_logit in zip(role_logits, head_logits, tail_logits):
            roles = set()
            for l, h, t in zip(*np.where(role_logit > threshold)):
                roles.add(tuple(self.hparams.id2label[l.item()]) + (h.item(), t.item()))
            links = set()
            for i1, (_, _, h1, t1) in enumerate(roles):
                for i2, (_, _, h2, t2) in enumerate(roles):
                    if i2 > i1:
                        # head_logits[i] 1,1,seq_len, seq_len
                        if head_logit[0, min(h1, h2), max(h1, h2)] > threshold:
                            if tail_logit[0, min(t1, t2), max(t1, t2)] > threshold:
                                links.add((h1, t1, h2, t2))
                                links.add((h2, t2, h1, t1))
            events = set()
            for e_label, sub_argus in groupby(sorted(roles), key=lambda x: x[0]):
                for event in clique_search(list(sub_argus), links):
                    role_ls = []
                    for argu in event:
                        role_label, start, end =argu[1], argu[2], argu[3]+1
                        if mapping is not None:
                            (start, end) = align_token_span((start, end), mapping)
                        role_ls.append(Role(start=start, end=end, label=role_label))                            
                    event = Event(label=e_label, roles=role_ls)
                    events.add(event)
            batch_events.append(events)
        return batch_events

    
    def configure_optimizers(self)  :
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.role_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr * 10},
            {'params': [p for n, p in self.role_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': self.hparams.lr * 10},
            {'params': [p for n, p in self.head_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.head_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.tail_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.tail_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': self.hparams.lr}
        ]
        
        optimizer = torch.optim.AdamW(grouped_params)
        scheduler_config = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler_config]


    def predict(self, text: str, device: str = 'cpu', threshold: Optional[float] = None):
        if threshold is None:
            threshold = self.hparams.threshold
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.hparams.max_length,
                                truncation=True,
                                return_offsets_mapping=True,
                                return_tensors='pt',
                                return_token_type_ids=False)
        mapping = inputs.pop('offset_mapping')
        mapping = mapping[0].tolist()
        self.freeze()
        self.to(device)
        inputs.to(device)
        role_logits, head_logits, tail_logits = self(**inputs)
        events = self.extract_events(role_logits, head_logits, tail_logits, threshold=threshold, mapping=mapping)
        return events