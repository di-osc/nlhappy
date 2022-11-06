from ...utils.make_model import PLMBaseModel, align_token_span
from ...layers.classifier import EfficientGlobalPointer, BiaffineSpanClassifier, GlobalPointer, EfficientBiaffineSpanClassifier
from ...layers.loss import MultiLabelCategoricalCrossEntropy, SparseMultiLabelCrossEntropy
from ...layers.dropout import MultiDropout
from ...metrics.event import Clique, CliqueF1
from ...metrics.span import SpanF1
import torch
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
            if host + argu in links:
                results.append(argu)
        return list(sorted(results))
    
    Argus = DedupList()
    for i1, (h1, t1) in enumerate(argus):
        for i2, (h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links and (h2, t2, h1, t1) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [tuple(sorted(argus))]

class GlobalPointerForCliqueExtraction(PLMBaseModel):
    def __init__(self, 
                 lr: float = 3e-5,
                 hidden_size: int = 64,
                 weight_decay: float = 0.01,
                 threshold: float = 0.0,
                 scheduler: str = 'linear_warmup',
                 **kwargs: Any) -> None:
        super().__init__()
        
        self.plm = self.get_plm_architecture()

        
        self.head_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=len(self.hparams.id2label),
                                                               add_rope=False,
                                                               tril_mask=False)
        self.tail_classifier = EfficientBiaffineSpanClassifier(input_size=self.plm.config.hidden_size,
                                                               hidden_size=hidden_size,
                                                               output_size=len(self.hparams.id2label),
                                                               add_rope=False,
                                                               tril_mask=False)
        
        self.dropout = MultiDropout()
        
        self.head_criterion = MultiLabelCategoricalCrossEntropy()
        self.tail_criterion = MultiLabelCategoricalCrossEntropy()
    
        self.val_metric = CliqueF1()
        self.train_head_metric = SpanF1()
        self.train_tail_metric = SpanF1()
        self.val_head_metric = SpanF1()
        self.val_tail_metric = SpanF1()
        

        
    def forward(self, input_ids, attention_mask):
        x = self.plm(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        head_logits = self.head_classifier(x, mask=attention_mask)
        tail_logits = self.tail_classifier(x, mask=attention_mask)
        return head_logits, tail_logits


    def step(self, batch):
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        head_logits, tail_logits = self(input_ids=input_ids, attention_mask=attention_mask)
        
        
        head_true = batch['head_ids']
        tail_true = batch['tail_ids']
        head_loss = self.head_criterion(head_logits.reshape(head_logits.shape[0] * head_logits.shape[1], -1), head_true.reshape(head_true.shape[0] * head_true.shape[1], -1))
        tail_loss = self.tail_criterion(tail_logits.reshape(tail_logits.shape[0]*tail_logits.shape[1], -1), tail_true.reshape(tail_true.shape[0]*tail_true.shape[1], -1))
        
        loss = (head_loss + tail_loss) / 2
        
        return loss, head_logits, tail_logits


    def training_step(self, batch, batch_idx):
        head_true = batch['head_ids']
        tail_true = batch['tail_ids']
        loss, head_logits, tail_logits = self.step(batch)
        head_pred = head_logits.gt(self.hparams.threshold)
        tail_pred = tail_logits.gt(self.hparams.threshold)
        self.train_head_metric(head_pred, head_true)
        self.train_tail_metric(tail_pred, tail_true)
        self.log('train/head_f1', self.train_head_metric, on_step=True, prog_bar=True, on_epoch=False)
        self.log('train/tail_f1', self.train_tail_metric, on_step=True, prog_bar=True, on_epoch=False)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        role_true, head_true, tail_true = batch['role_ids'], batch['head_ids'], batch['tail_ids']
        loss, head_logits, tail_logits = self.step(batch)
        batch_true_cliques = self.extract_cliques(role_true, head_true, tail_true)
        batch_pred_cliques = self.extract_cliques(role_true, head_logits, tail_logits)
        self.val_metric(batch_pred_cliques, batch_true_cliques)
        self.val_head_metric(head_logits.gt(self.hparams.threshold), head_true)
        self.val_tail_metric(tail_logits.gt(self.hparams.threshold), tail_true)
        self.log('val/f1', self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/head', self.val_head_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/tail', self.val_tail_metric, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}


    def extract_cliques(self, role_logits: torch.Tensor, head_logits: torch.Tensor, tail_logits: torch.Tensor, threshold: Union[None, float]= None, mapping: Optional[List[Tuple]]= None):
        if threshold is None:
            threshold = self.hparams.threshold
        role_logits = role_logits.cpu().detach().numpy()
        head_logits = head_logits.cpu().detach().numpy()
        tail_logits = tail_logits.cpu().detach().numpy()
        assert len(head_logits) == len(tail_logits) == len(role_logits)
        batch_cliques = []
        for role_logit, head_logit, tail_logit in zip(role_logits, head_logits, tail_logits):
            roles = DedupList()
            for _, h, t in zip(*np.where(role_logit > threshold)):
                roles.append((h.item(), t.item()))
            links = set()
            for i1, (h1, t1) in enumerate(roles):
                for i2, (h2, t2) in enumerate(roles):
                    # if i2 > i1:
                    if i2 != i1:
                        # head_logits[i] 1,1,seq_len, seq_len
                        if head_logit[0, h1, h2] > threshold and tail_logit[0, t1, t2] > threshold:
                                links.add((h1, t1, h2, t2))
            hosts_ls = clique_search(roles, links)
            cliques = set()
            for hosts in hosts_ls:
                cliques.add(Clique(hosts=hosts))
            batch_cliques.append(cliques)
        return batch_cliques

    
    def configure_optimizers(self)  :
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(grouped_params, lr=self.hparams.lr)
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