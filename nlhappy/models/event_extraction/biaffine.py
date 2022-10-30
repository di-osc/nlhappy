from ...utils.make_model import PLMBaseModel
from ...layers.classifier import EfficientBiaffineSpanClassifier
from ...layers.loss import MultiLabelCategoricalCrossEntropy
from ...layers.dropout import MultiDropout
from ...metrics.event import EventF1, Event, Role
from ...metrics.span import SpanF1
import torch
from itertools import groupby
from typing import Any, Union
import numpy as np


class DedupList(list):
    """定义去重的list
    """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def clique_search(argus, links: set):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    def neighbors(host, argus, links):
        """构建邻集（host节点与其所有邻居的集合）
        """
        results = [host]
        for argu in argus:
            if host[1:] + argu[1:] in links:
                results.append(argu)
        return list(sorted(results))
    
    Argus = DedupList()
    for i1, (_, h1, t1) in enumerate(argus):
        for i2, (_, h2, t2) in enumerate(argus):
            if i2 > i1:
                # 找出不相邻的节点
                if (h1, t1, h2, t2) not in links:
                    # 然后找到跟他们有关系的节点
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

class BiaffineForEventExtraction(PLMBaseModel):
    def __init__(self, 
                 lr: float = 3e-5,
                 hidden_size: int = 64,
                 weight_decay: float = 0.01,
                 threshold: float = 0.0,
                 scheduler: str = 'linear_warmup',
                 **kwargs: Any) -> None:
        super().__init__()
        
        self.plm = self.get_plm_architecture()

        self.ent_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=len(self.hparams.label2id))
        
        self.head_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=1,
                                                               add_rope=False)
        
        self.tail_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=1,
                                                               add_rope=False)
        
        self.dropout = MultiDropout()
        
        self.ent_criterion = MultiLabelCategoricalCrossEntropy()
        self.head_criterion = MultiLabelCategoricalCrossEntropy()
        self.tail_criterion = MultiLabelCategoricalCrossEntropy()
    
        self.event_metric = EventF1()             
        self.ent_metric = SpanF1()
        self.head_metric = SpanF1()
        self.tail_metric = SpanF1()

        
    def forward(self, input_ids, attention_mask):
        x = self.plm(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        ent_logits = self.ent_classifier(x, mask=attention_mask)
        head_logits = self.head_classifier(x, mask=attention_mask)
        tail_logits = self.tail_classifier(x, mask=attention_mask)
        return ent_logits, head_logits, tail_logits


    def step(self, batch):
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        ent_logits, head_logits, tail_logits = self(input_ids, attention_mask)
        
        ent_true = batch['ent_tags']
        head_true = batch['head_tags']
        tail_true = batch['tail_tags']
        
        role_loss = self.ent_criterion(ent_logits.reshape(ent_logits.shape[0]*ent_logits.shape[1], -1), ent_true.reshape(ent_true.shape[0]*ent_true.shape[1], -1))
        head_loss = self.head_criterion(head_logits.reshape(head_logits.shape[0]*head_logits.shape[1], -1), head_true.reshape(head_true.shape[0]*head_true.shape[1], -1))
        tail_loss = self.tail_criterion(tail_logits.reshape(tail_logits.shape[0]*tail_logits.shape[1], -1), tail_true.reshape(tail_true.shape[0]*tail_true.shape[1], -1))
        
        loss = (role_loss + head_loss + tail_loss) / 3
        
        return loss, ent_logits, head_logits, tail_logits


    def training_step(self, batch, batch_idx):
        loss, ent_logits, head_logits, tail_logits = self.step(batch)
        ent_pred = ent_logits.ge(self.hparams.threshold)
        head_pred = head_logits.ge(self.hparams.threshold)
        tail_pred = tail_logits.ge(self.hparams.threshold)
        ent_true = batch['ent_tags']
        head_true = batch['head_tags']
        tail_true = batch['tail_tags']
        self.ent_metric(ent_pred, ent_true)
        self.head_metric(head_pred, head_true)
        self.tail_metric(tail_pred, tail_true)
        self.log('train/ent_f1', self.ent_metric, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/head_f1', self.head_metric, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/tail_f1', self.tail_metric, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_f1', loss)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        role_true, head_true, tail_true = batch['ent_tags'], batch['head_tags'], batch['tail_tags']
        loss, role_logits, head_logits, tail_logits = self.step(batch)
        batch_true_events = self.extract_events(role_true, head_true, tail_true)
        batch_events = self.extract_events(role_logits, head_logits, tail_logits)
        self.event_metric(batch_events, batch_true_events)
        self.log('val/f1', self.event_metric, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}


    def extract_events(self, role_logits: torch.Tensor, head_logits: torch.Tensor, tail_logits: torch.Tensor, threshold: Union[None, float]=None):
        if threshold is None:
            threshold = self.hparams.threshold
        role_logits = role_logits.cpu().detach().numpy()
        head_logits = head_logits.cpu().detach().numpy()
        tail_logits = tail_logits.cpu().detach().numpy()
        assert len(head_logits) == len(tail_logits) == len(role_logits)
        batch_events = []
        for rs, hs, ts in zip(role_logits, head_logits, tail_logits):
            roles = set()
            for l, h, t in zip(*np.where(rs > threshold)):
                roles.add((l, h, t))
            events = set()
            for _, sub_roles in groupby(roles, key=lambda x: x[0]):
                links = set()
                sub_roles = list(sub_roles)
                for  i, (e_label, h1, t1) in enumerate(sub_roles):
                    for j, (e_label, h2, t2) in enumerate(sub_roles):
                        if j > i:
                            if hs[0][h1][h2]>0 and ts[0][t1][t2]>0:
                                links.add((h1, t1, h2, t2))
                                links.add((h2, t2, h1, t1))
                event_ls = clique_search(argus=list(sub_roles), links=links)
                for e in event_ls:
                    e_label = self.hparams.id2label[int(e[0][0])]
                    events.add(Event(label=e_label, roles=set([Role(start=r[1], end=r[1], label=None) for r in e])))
            batch_events.append(events)
        return batch_events

    
    def configure_optimizers(self)  :
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.ent_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr * 5},
            {'params': [p for n, p in self.ent_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': self.hparams.lr * 5},
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


    def predict(self, text: str, device: str = 'cpu'):
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.hparams.max_length,
                                truncation=True,
                                return_offsets_mapping=True,
                                return_tensors='pt',
                                return_token_type_ids=False)
        mapping = inputs.pop('offset_mapping')
        mapping = mapping[0].tolist()
        inputs.to(device)
        role_logits, head_logits, tail_logits = self(**inputs)
        events = self.extract_events(role_logits, head_logits, tail_logits)
        return events