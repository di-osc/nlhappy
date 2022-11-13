from ...layers import MultiDropout, EfficientBiaffineSpanClassifier
from ...layers.loss import SparseMultiLabelCrossEntropy
from ...metrics.triple import TripleF1, Triple
from ...metrics.span import SpanF1
import torch
from torch import Tensor
from typing import List, Set
from ...utils.make_model import align_token_span, PLMBaseModel
from typing import Optional


class BLinkerForEntityRelationExtraction(PLMBaseModel):
    """基于biaffine的实体关系联合抽取模型
    """
    def __init__(self,
                 hidden_size: int = 64,
                 lr: float = 3e-5,
                 scheduler: str = 'linear_warmup', 
                 weight_decay: float = 0.01,
                 threshold: float = 0.0,
                 **kwargs):
        super().__init__()
        
        self.plm = self.get_plm_architecture()
        self.dropout = MultiDropout()       
        # 主语 宾语分类器
        self.ent_classifier = EfficientBiaffineSpanClassifier(self.plm.config.hidden_size, hidden_size, len(self.hparams.id2combined))
        # 主语 宾语 头对齐
        self.head_classifier = EfficientBiaffineSpanClassifier(self.plm.config.hidden_size, hidden_size, len(self.hparams.id2rel), add_rope=False, tril_mask=False)
        # 主语 宾语 尾对齐
        self.tail_classifier = EfficientBiaffineSpanClassifier(self.plm.config.hidden_size, hidden_size, len(self.hparams.id2rel), add_rope=False, tril_mask=False)

        self.ent_criterion = SparseMultiLabelCrossEntropy()
        self.head_criterion = SparseMultiLabelCrossEntropy()
        self.tail_criterion = SparseMultiLabelCrossEntropy()
        

        self.ent_metric = SpanF1()
        self.head_metric = SpanF1()
        self.tail_metric = SpanF1()
        self.val_metric = TripleF1()

    

    def forward(self, input_ids, attention_mask=None):
        hidden_state = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden_state = self.dropout(hidden_state)
        ent_logits = self.ent_classifier(hidden_state, mask=attention_mask)
        head_logits = self.head_classifier(hidden_state, mask=attention_mask)
        tail_logits = self.tail_classifier(hidden_state, mask=attention_mask)
        return ent_logits, head_logits, tail_logits


    def shared_step(self, batch, is_train: bool):
        #inputs为bert常规输入, span_ids: [batch_size, 2, seq_len, seq_len],
        #head_ids: [batch_size, len(label2id), seq_len, seq_len], tail_ids: [batch_size, len(label2id), seq_len, seq_len]
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        ent_true, head_true, tail_true = batch['combined_tags'],  batch['head_tags'], batch['tail_tags']
        ent_logits, head_logits, tail_logits = self(input_ids=input_ids, attention_mask=attention_mask)
        
        if is_train:
            b,c,s,s = ent_logits.shape
            ent_logits = ent_logits.reshape(b,c,s*s)
            ent_true = ent_true[..., 0] * s + ent_true[..., 1]
            ent_loss = self.ent_criterion(ent_logits, ent_true)
        
            b,c,s,s = head_logits.shape
            head_logits = head_logits.reshape(b,c,s*s)
            head_true = head_true[..., 0] * s + head_true[..., 1]
            head_loss = self.head_criterion(head_logits, head_true)
            
            b,c,s,s = tail_logits.shape
            tail_logits = tail_logits.reshape(b,c,s*s)
            tail_true = tail_true[..., 0] * s + tail_true[..., 1]
            tail_loss = self.tail_criterion(tail_logits, tail_true)
            self.log('train/ent_loss', ent_loss, prog_bar=True, on_step=True)
            self.log('train/head_loss', head_loss, prog_bar=True, on_step=True)
            self.log('train/tail_loss', tail_loss, prog_bar=True, on_step=True)
            
            loss = sum([ent_loss, head_loss, tail_loss]) / 3
            return loss
        else:
            return ent_logits, head_logits, tail_logits, ent_true, head_true, tail_true

        
    def training_step(self, batch, batch_idx) -> dict:
        # 训练阶段不进行解码, 会比较慢
        loss = self.shared_step(batch, is_train=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx) -> dict:
        ent_logits, head_logits, tail_logits, ent_true, head_true, tail_true = self.shared_step(batch, is_train=False)
        
        self.ent_metric(ent_logits, ent_true)
        self.log('val/ent', self.ent_metric, on_epoch=True, prog_bar=True)

        self.head_metric(head_logits, head_true)
        self.log('val/head', self.head_metric, on_epoch=True, prog_bar=True)

        self.tail_metric(tail_logits, tail_true)
        self.log('val/tail', self.tail_metric, on_epoch=True, prog_bar=True)
        
        batch_triples = self.extract_triple(ent_logits, head_logits, tail_logits, threshold=self.hparams.threshold)
        batch_true_triples = self.extract_triple(ent_true, head_true, tail_true, threshold=self.hparams.threshold)
        self.val_metric(batch_triples, batch_true_triples)
        self.log('val/f1', self.val_metric, on_epoch=True, prog_bar=True)


    def configure_optimizers(self)  :
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.hparams.lr)
        scheduler_config = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler_config]



    def extract_triple(self, 
                       ent_logits: Tensor,
                       head_logits: Tensor, 
                       tail_logtis: Tensor, 
                       threshold: Optional[float] = None) -> List[Set[Triple]]:
        """
        将三个globalpointer预测的结果进行合并，得到三元组的预测结果
        参数:
        - ent_logits: [batch_size, 2, seq_len, seq_len]
        - head_logits: [batch_size, predicate_type, seq_len, seq_len]
        - tail_logtis: [batch_size, predicate_type, seq_len, seq_len]
        返回:
        - batch_size大小的列表，每个元素是一个集合，集合中的元素是三元组
        """
        if threshold is None:
            threshold = self.hparams.threshold
        ent_logits = ent_logits.chunk(ent_logits.shape[0])
        head_logits = head_logits.chunk(head_logits.shape[0])
        tail_logtis = tail_logtis.chunk(tail_logtis.shape[0])
        assert len(head_logits) == len(tail_logtis) == len(ent_logits)
        batch_triples = []
        for i in range(len(ent_logits)):
            subjects, objects = set(), set()
            for l, h, t in zip(*torch.where(ent_logits[i].squeeze(0) > threshold)):
                combined_label = self.hparams.id2combined[l.item()]
                if combined_label[0] == '主体':
                    subjects.add((h, t, combined_label[1])) 
                else:
                    objects.add((h, t, combined_label[1]))
            
            triples = set()
            for sh, st, sl, in subjects:
                for oh, ot, ol in objects:
                    p1s = torch.where(head_logits[i].squeeze(0)[:, sh, oh] > threshold)[0].tolist()
                    p2s = torch.where(tail_logtis[i].squeeze(0)[:, st, ot] > threshold)[0].tolist()
                    ps = set(p1s) & set(p2s)
                    if len(ps) > 0:
                        for p in ps:
                            triples.add(Triple(triple=(sh.item(), st.item(), self.hparams.id2rel[p], oh.item(), ot.item())))
            batch_triples.append(triples)
        return batch_triples
            
        
    def predict(self, text: str, device:str='cpu', threshold = None) -> Set[Triple]:
        """模型预测
        参数:
        - text: 要预测的单条文本
        - device: 设备
        - threshold: 三元组抽取阈值, 如果为None, 则为模型训练时的阈值
        返回
        - 预测的三元组
        """
        max_length = min(len(text), self.hparams.max_length)
        inputs = self.tokenizer(
                text, 
                padding='max_length',  
                max_length=max_length,
                return_tensors='pt',
                return_token_type_ids=False,
                truncation=True)
        inputs.to(torch.device(device))
        ent_logits, head_logits, tail_logits = self(**inputs)
        if threshold == None:
            batch_triples = self.extract_triple(ent_logits, head_logits, tail_logits, threshold=self.hparams.threshold)
        else:
            batch_triples = self.extract_triple(ent_logits, head_logits, tail_logits, threshold=threshold)
        rels = []
        if len(batch_triples) >0:
            triples = [(triple[0], triple[1], triple[2], triple[3], triple[4])  for s in batch_triples for triple in s]
            offset_mapping = self.tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True)['offset_mapping']
            for triple in triples:
                sub = align_token_span((triple[0], triple[1]+1), offset_mapping)
                obj = align_token_span((triple[3], triple[4]+1), offset_mapping)
                rels.append((sub[0],sub[1],triple[2],obj[0],obj[1]))
        return rels