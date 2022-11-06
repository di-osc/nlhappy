from ...utils.make_model import PLMBaseModel, align_token_span
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
    for i1, (h1, t1) in enumerate(argus):
        for i2, (h2, t2) in enumerate(argus):
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

        self.role_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=len(self.hparams.id2event))
        
        self.head_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=len(self.hparams.id2event),
                                                               add_rope=False)
        
        self.tail_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=len(self.hparams.id2event),
                                                               add_rope=False)
        
        self.dropout = MultiDropout()
        
        self.ent_criterion = MultiLabelCategoricalCrossEntropy()
        self.head_criterion = MultiLabelCategoricalCrossEntropy()
        self.tail_criterion = MultiLabelCategoricalCrossEntropy()
    
        self.event_metric = EventF1()             
        self.role_metric = SpanF1()
        self.head_metric = SpanF1()
        self.tail_metric = SpanF1()

        
    def forward(self, input_ids, attention_mask):
        x = self.plm(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        ent_logits = self.role_classifier(x, mask=attention_mask)
        head_logits = self.head_classifier(x, mask=attention_mask)
        tail_logits = self.tail_classifier(x, mask=attention_mask)
        return ent_logits, head_logits, tail_logits


    def step(self, batch):
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        role_logits, head_logits, tail_logits = self(input_ids, attention_mask)
        
        role_true = batch['role_tags']
        head_true = batch['head_tags']
        tail_true = batch['tail_tags']
        
        role_loss = self.ent_criterion(role_logits.reshape(role_logits.shape[0]*role_logits.shape[1], -1), role_true.reshape(role_true.shape[0]*role_true.shape[1], -1))
        head_loss = self.head_criterion(head_logits.reshape(head_logits.shape[0]*head_logits.shape[1], -1), head_true.reshape(head_true.shape[0]*head_true.shape[1], -1))
        tail_loss = self.tail_criterion(tail_logits.reshape(tail_logits.shape[0]*tail_logits.shape[1], -1), tail_true.reshape(tail_true.shape[0]*tail_true.shape[1], -1))
        
        loss = (role_loss + head_loss + tail_loss) / 3
        
        return loss, role_logits, head_logits, tail_logits, role_true, head_true, tail_true


    def training_step(self, batch, batch_idx):
        loss, ent_logits, head_logits, tail_logits, role_true, head_true, tail_true = self.step(batch)
        role_pred = ent_logits.gt(self.hparams.threshold)
        head_pred = head_logits.gt(self.hparams.threshold)
        tail_pred = tail_logits.gt(self.hparams.threshold)
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
        return {'val_loss': loss}


    def extract_events(self, role_logits: torch.Tensor, head_logits: torch.Tensor, tail_logits: torch.Tensor, threshold: Union[None, float]=None, mapping = None):
        if threshold is None:
            threshold = self.hparams.threshold
        role_logits = role_logits.cpu().detach().numpy()
        head_logits = head_logits.cpu().detach().numpy()
        tail_logits = tail_logits.cpu().detach().numpy()
        assert len(head_logits) == len(tail_logits) == len(role_logits)
        batch_events = []
        # 遍历batch内的每一组
        for rs, hs, ts in zip(role_logits, head_logits, tail_logits):
            # 找到所有的role
            events = set()
            roles_per_event = [set() for _ in range(len(self.hparams.id2event))]
            for l, h, t in zip(*np.where(rs > threshold)):
                roles_per_event[l].add((h, t))
            for i, roles in enumerate(roles_per_event):
                if roles:
                    e_label = self.hparams.id2event[i]
                    links = set()
                    true_roles = set()
                    for r1 in roles:
                        for r2 in roles:
                            if hs[i][r1[0]][r2[0]] > self.hparams.threshold and ts[i][r1[1]][r2[1]] > self.hparams.threshold:
                                links.add((r1[0], r1[1], r2[0], r2[1]))
                                links.add((r2[0], r2[1], r1[0], r1[1]))
                                true_roles.add(r1)
                                true_roles.add(r2)
                    for event in clique_search(list(true_roles), links):
                        role_ls = []
                        for argu in event:
                            start, end =argu[0], argu[1]+1
                            if mapping is not None:
                                (start, end) = align_token_span((start, end), mapping)
                            role_ls.append(Role(start=start, end=end, label=None))                            
                        event = Event(label=e_label, roles=role_ls)
                        events.add(event)
            batch_events.append(events)
        return batch_events

    
    def configure_optimizers(self)  :
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
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