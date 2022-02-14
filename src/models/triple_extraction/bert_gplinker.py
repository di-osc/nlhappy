import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from ...layers import EfficientGlobalPointer
from ...layers.loss import MultiLabelCategoricalCrossEntropy, SparseMultiLabelCrossEntropy
from ...metrics.triple import TripleF1
import torch
from torch import Tensor
from typing import List, Set
from ...utils.type import Triple





class BertGPLinker(pl.LightningModule):
    """基于globalpointer的关系抽取模型
    参考:
    - https://kexue.fm/archives/8888
    - https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction_gplinker.py
    """
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        dropout: float,
        weight_decay: float,
        threshold: float =0.0,
        **data_params
    ):
        super(BertGPLinker, self).__init__()
        self.save_hyperparameters()
        
        self.label2id = data_params['label2id']
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])

        # 主语 宾语分类器
        self.span_classifier = EfficientGlobalPointer(self.bert.config.hidden_size, hidden_size, 2)  # 0: suject  1: object
        # 主语 宾语 头对齐
        self.head_classifier = EfficientGlobalPointer(self.bert.config.hidden_size, hidden_size, len(data_params['label2id']), RoPE=False, tril_mask=False)
        # 主语 宾语 尾对齐
        self.tail_classifier = EfficientGlobalPointer(self.bert.config.hidden_size, hidden_size, len(data_params['label2id']), RoPE=False, tril_mask=False)

        self.span_criterion = MultiLabelCategoricalCrossEntropy()
        self.head_criterion = MultiLabelCategoricalCrossEntropy()
        self.tail_criterion = MultiLabelCategoricalCrossEntropy()

        self.train_f1 = TripleF1()
        self.val_f1 = TripleF1()
        self.test_f1 = TripleF1()

    def forward(self, inputs):
        hidden_state = self.bert(**inputs).last_hidden_state
        span_logits = self.span_classifier(hidden_state, mask=inputs['attention_mask'])
        head_logits = self.head_classifier(hidden_state, mask=inputs['attention_mask'])
        tail_logits = self.tail_classifier(hidden_state, mask=inputs['attention_mask'])
        return span_logits, head_logits, tail_logits


    def shared_step(self, batch):
        #inputs为bert常规输入, span_ids: [batch_size, 2, seq_len, seq_len],
        #head_ids: [batch_size, len(label2id), seq_len, seq_len], tail_ids: [batch_size, len(label2id), seq_len, seq_len]
        inputs, span_true, head_true, tail_true = batch['inputs'], batch['span_ids'], batch['head_ids'], batch['tail_ids']
        span_logits, head_logits, tail_logits = self(inputs)

        span_logits_ = span_logits.reshape(span_logits.shape[0] * span_logits.shape[1], -1)
        span_true_ = span_true.reshape(span_true.shape[0] * span_true.shape[1], -1)
        span_loss = self.span_criterion(span_logits_, span_true_)
        
        head_logits_ = head_logits.reshape(head_logits.shape[0] * head_logits.shape[1], -1)
        head_true_ = head_true.reshape(head_true.shape[0] * head_true.shape[1], -1)
        head_loss = self.head_criterion(head_logits_, head_true_)
        
        tail_logits_ = tail_logits.reshape(tail_logits.shape[0] * tail_logits.shape[1], -1)
        tail_true_ = tail_true.reshape(tail_true.shape[0] * tail_true.shape[1], -1)
        tail_loss = self.tail_criterion(tail_logits_, tail_true_)
        
        loss = span_loss + head_loss + tail_loss
        
        return loss, span_logits, head_logits, tail_logits

    def training_step(self, batch, batch_idx):
        loss, span_logits, head_logits, tail_logits = self.shared_step(batch)
        # self.train_f1(batch_triples, batch_true_triples)
        # self.log('train/f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        inputs, span_true, head_true, tail_true = batch['inputs'], batch['span_ids'], batch['head_ids'], batch['tail_ids']
        loss, span_logits, head_logits, tail_logits = self.shared_step(batch)
        batch_triples = self.extract_triple(span_logits, head_logits, tail_logits)
        batch_true_triples = self.extract_triple(span_true, head_true, tail_true)
        self.val_f1(batch_triples, batch_true_triples)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss, batch_triples, batch_true_triples = self.shared_step(batch)
        self.test_f1(batch_triples, batch_true_triples)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss}

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
            {'params': [p for n, p in self.span_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr* 100, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.span_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr* 100, 'weight_decay': 0.0},
            {'params': [p for n, p in self.head_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr* 100, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.head_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr* 100, 'weight_decay': 0.0},
            {'params': [p for n, p in self.tail_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr* 100, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.tail_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr* 100, 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(grouped_parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]

    def extract_triple(
        self,
        span_logits: Tensor, 
        head_logits: Tensor, 
        tail_logtis: Tensor, 
        threshold: float=0.0
        ) -> List[Set[Triple]]:
        """
        将三个globalpointer预测的结果进行合并，得到三元组的预测结果
        参数:
        - span_logits: [batch_size, 2, seq_len, seq_len]
        - head_logits: [batch_size, predicate_type, seq_len, seq_len]
        - tail_logtis: [batch_size, predicate_type, seq_len, seq_len]
        返回:
        - batch_size大小的列表，每个元素是一个集合，集合中的元素是三元组
        """
        span_logits = span_logits.chunk(span_logits.shape[0])
        head_logits = head_logits.chunk(head_logits.shape[0])
        tail_logtis = tail_logtis.chunk(tail_logtis.shape[0])
        assert len(span_logits) == len(head_logits) == len(tail_logtis)
        batch_triples = []
        for i in range(len(span_logits)):
            subjects, objects = set(), set()
            for l, h, t in zip(*torch.where(span_logits[i].squeeze(0) > threshold)):
                if l == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))
            
            triples = set()
            for sh, st in subjects:
                for oh, ot in objects:
                    p1s = torch.where(head_logits[i].squeeze(0)[:, sh, oh] > threshold)[0].tolist()
                    p2s = torch.where(tail_logtis[i].squeeze(0)[:, st, ot] > threshold)[0].tolist()
                    ps = set(p1s) & set(p2s)
                    if len(ps) > 0:
                        for p in ps:
                            triples.add(Triple(triple=(sh.item(), sh.item(), p, oh.item(), ot.item())))
            batch_triples.append(triples)
        return batch_triples
    


