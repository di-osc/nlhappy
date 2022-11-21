import torch.nn as nn
import torch


class OneRelSpanClassifier(nn.Module):
    """枚举出sel_len * sel_len个span,并进行分类

    Args:
        input_size(int): 输入特征维度
        output_size(int): 分类的类别数量
    输出形状: [batch, seq, seq, output_size]
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 ):
        super().__init__()
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size*2, input_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(input_size, output_size)
        self.layernorm = nn.LayerNorm(input_size)
        
    def forward(self, x): 
        batch_size, seq_len, hidden_size = x.size()
        # head: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,1,1, 2,2,2, 3,3,3
        head = x.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size).reshape(batch_size, seq_len*seq_len, hidden_size)
        # tail: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,2,3, 1,2,3, 1,2,3
        tail = x.repeat(1, seq_len, 1)
        # 两两token的特征拼接在一起
        pairs = torch.cat([head, tail], dim=-1)
        pairs = self.linear1(pairs)
        pairs = self.layernorm(pairs)
        pairs = self.activation(pairs)
        scores = self.linear2(pairs).reshape(batch_size, seq_len, seq_len, self.output_size)
        return scores