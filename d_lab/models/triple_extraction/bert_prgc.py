import pytorch_lightning as pl
from transformers import BertModel
from ...layers.classifier import SimpleDense
import torch.nn as nn

class BertPRGC(pl.LightningModule):
    '''实体关系联合抽取模型PRGC
    参考:
    - https://github.com/hy-struggle/PRGC
    - https://github.com/xiangking/ark-nlp/blob/main/ark_nlp/model/re/prgc_bert/prgc_bert.py
    '''
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        weight_decay: float,
        dropout: float,
        **data_params
    ):
        super(BertPRGC, self).__init__()
        self.save_hyperparameters()

        self.label2id = data_params['label2id']
        self.id2label = {v:k for k,v in self.label2id.items()}

        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.rel_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(self.label2id))
        self.rel_embedding = nn.Embedding(len(self.label2id), hidden_size)

    
    def forward(self, inputs):
        pooler_output = self.bert(**inputs).pooler_output
        # 关系分类是一个多标签分类, 得到文本所有包含的关系
        rel_logits = self.rel_classifier(pooler_output)



