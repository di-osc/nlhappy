from ...utils.make_model import PLMBaseModel
from ...layers.classifier import GlobalPointer, EfficientGlobalPointer
from ...layers.loss import MultiLabelCategoricalCrossEntropy, SparseMultiLabelCrossEntropy
from ...layers.dropout import MultiDropout
from ...metrics.event import EventF1, Event, Role
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
                 scheduler: str = 'harmonic_epoch',
                 **kwargs: Any) -> None:
        super().__init__()
        self.plm = self.get_plm_architecture()

        self.role_classifier = EfficientGlobalPointer(input_size=self.plm.config.hidden_size,
                                                      hidden_size=hidden_size,
                                                      output_size=len(self.hparams.label2id),
                                                      RoPE=True,
                                                      tril_mask=True)
        self.head_classifier = EfficientGlobalPointer(input_size=self.plm.config.hidden_size,
                                                      hidden_size=hidden_size,
                                                      output_size=1,
                                                      RoPE=False)
        self.tail_classifier = EfficientGlobalPointer(input_size=self.plm.config.hidden_size,
                                                      hidden_size=hidden_size,
                                                      output_size=1,
                                                      RoPE=False)
        self.dropout = MultiDropout()
        # self.role_criterion = MultiLabelCategoricalCrossEntropy()
        # self.head_criterion = MultiLabelCategoricalCrossEntropy()
        # self.tail_criterion = MultiLabelCategoricalCrossEntropy()
        self.role_criterion = SparseMultiLabelCrossEntropy()
        self.head_criterion = SparseMultiLabelCrossEntropy()
        self.tail_criterion = SparseMultiLabelCrossEntropy()
        self.val_metric = EventF1()                  

        
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
        
        role_true = batch['role_ids']
        head_true = batch['head_ids']
        tail_true = batch['tail_ids']
        
        # role_logits_ = role_logits.reshape(role_logits.shape[0] * role_logits.shape[1], -1)
        # role_true_ = role_true.reshape(role_true.shape[0] * role_true.shape[1], -1)
        # role_loss = self.role_criterion(role_logits_, role_true_)
        role_loss = self.role_criterion(role_logits, role_true)
        
        # head_logits_ = head_logits.reshape(head_logits.shape[0] * head_logits.shape[1], -1)
        # head_true_ = head_true.reshape(head_true.shape[0] * head_true.shape[1], -1)
        # head_loss = self.head_criterion(head_logits_, head_true_)
        head_loss = self.head_criterion(head_logits, head_true)
        
        # tail_logits_ = tail_logits.reshape(tail_logits.shape[0] * tail_logits.shape[1], -1)
        # tail_true_ = tail_true.reshape(tail_true.shape[0] * tail_true.shape[1], -1)
        # tail_loss = self.tail_criterion(tail_logits_, tail_true_)
        tail_loss = self.tail_criterion(tail_logits, tail_true)
        
        loss = sum([role_loss, head_loss, tail_loss]) / 3
        return loss, role_logits, head_logits, tail_logits


    def training_step(self, batch, batch_idx):
        loss, role_logits, head_logits, tail_logits = self.step(batch)
        self.log('train/loss', loss)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        role_true, head_true, tail_true = batch['role_ids'], batch['head_ids'], batch['tail_ids']
        loss, role_logits, head_logits, tail_logits = self.step(batch)
        batch_events = self.extract_events(role_logits, head_logits, tail_logits)
        batch_true_events = self.extract_events(role_true, head_true, tail_true)
        self.val_metric(batch_events, batch_true_events)
        self.log('val/f1', self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}


    def extract_events(self, role_logits: torch.Tensor, head_logits: torch.Tensor, tail_logits: torch.Tensor, threshold: Union[None, float]=None):
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
                        role_ls.append(Role(start=start, end=end, label=role_label))
                    event = Event(label=e_label, roles=role_ls)
                    events.add(event)
            batch_events.append(events)
        return batch_events

    
    def configure_optimizers(self)  :
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_params)
        scheduler_config = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler_config]

    def predict(self):
        pass