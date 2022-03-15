import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from ...layers import SimpleDense, EfficientGlobalPointer
import torch.nn as nn
import torch


class BertRelinker(pl.LightningModule):
    """GPLinker的变体
    思路:
    - 先识别出文本中包含的关系
    - 识别主语, 宾语
    - 主宾头对齐
    - 主宾尾对齐
    """
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        dropout: float,
        weight_decay: float,
        threshold: float = 0.5,
        **data_params
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bert = BertModel.from_pretrained(self.hparams['pretrained_dir'] + self.hparams['pretrained_model'])
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams['pretrained_dir'] + self.hparams['pretrained_model'])

        self.rel_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(self.hparams['label2id']))
        self.span_classifier = EfficientGlobalPointer(self.bert.config.hidden_size, hidden_size, 2)
        self.head_classifier = EfficientGlobalPointer(self.bert.config.hidden_size, hidden_size, 1)
        self.tail_classifier = EfficientGlobalPointer(self.bert.config.hidden_size, hidden_size, 1)
        self.rel_embedding = nn.Embedding(len(self.hparams['label2id']), hidden_size)


    def forward(self, inputs):
        hidden_state = self.bert(**inputs).last_hidden_state
        pooled = self.bert(**inputs).pooler_output
        rel_logits = self.rel_classifier(pooled)
        so_logits = []
        for logits in rel_logits:
            embeds = self.rel_embedding(torch.nonezero(logits.sigmoid().ge(self.hparams['threshold']).reshape(-1)))
            so_logit = []
            for embed in embeds:
                hidden_state = (hidden_state + embed) / 2
                span_logits = self.span_classifier(hidden_state)
                head_logits = self.head_classifier(hidden_state)
                tail_logits = self.tail_classifier(hidden_state)
                so_logit.append((span_logits, head_logits, tail_logits))
            so_logits.append(so_logit)
        return rel_logits, so_logits


    def shared_step(self, batch):
        inputs, span_true, head_true, tail_true = batch['inputs'], batch['span_ids'], batch['head_ids'], batch['tail_ids']
        rel_logits, so_logits = self.forward(inputs)


        
        
