import torch.nn as nn
import torch


class OneRelClassifier(nn.Module):
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tag_size: int,
                 dropout_prob: float):
        super().__init__()
        self.output_size = output_size
        self.tag_size =tag_size
        self.linear1 = nn.Linear(input_size*2, input_size*3)
        self.dropout =  nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(input_size*3, tag_size*output_size)
        
    def forward(self, x): 
        batch_size, seq_len, hidden_size = x.size()
        # head: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,1,1, 2,2,2, 3,3,3
        head = x.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size).reshape(batch_size, seq_len*seq_len, hidden_size)
        # tail: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,2,3, 1,2,3, 1,2,3
        tail = x.repeat(1, seq_len, 1)
        # 两两token的特征拼接在一起
        pairs = torch.cat([head, tail], dim=-1)
        pairs = self.linear1(pairs)
        pairs = self.dropout(pairs)
        pairs = self.activation(pairs)
        scores = self.linear2(pairs).reshape(batch_size, seq_len, seq_len, self.output_size, self.tag_size)
        return scores
        
