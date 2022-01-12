from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from typing import Optional
import torch

class BERTCrossEncoder(nn.Module):
    '''
    - bert_encoder: BERT模型
    - mid_size: 隐藏层大小
    - output_size: 输出大小
    '''
    def __init__(
        self, 
        encoder_name: str,
        mid_size: int,
        output_size: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_name)
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, mid_size)
        self.classifier = nn.Linear(mid_size, output_size)

    def forward(self, inputs):
        x = self.bert(**inputs).last_hidden_state
        x = x.mean(dim=1)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits


class BERTBiEncoder(nn.Module):
    '''
    句子对分类模型
    '''
    def __init__(
        self, 
        encoder_name: str,
        mid_size: int,
        output_size: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_name)
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size*3, mid_size)
        self.classifier = nn.Linear(mid_size, output_size)

    def forward(self, inputs):
        inputs_a, inputs_b = inputs
        encoded_a = self.bert(**inputs_a).pooler_output
        encoded_b = self.bert(**inputs_b).pooler_output
        abs_diff = torch.abs(encoded_a - encoded_b)
        concat = torch.cat((encoded_a, encoded_b, abs_diff), dim=-1)
        hidden = F.relu(self.hidden_layer(concat))
        logits = self.classifier(hidden)
        return logits
