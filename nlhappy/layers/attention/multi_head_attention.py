import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import math

# 乘性的注意力算法
def scaled_mul_self_attention(query, key, value, mask=None):
    """
    # 为什么scaled(也就是除以sqrt(d_k))？: 
    - 1. 维度升高使得乘性注意力机制的方差变大
    - 2. 进而出现极大值使得softmax梯度消失
    - 3. 通过scale控制方差,进而稳定梯度流, 防止梯度爆炸
    - https://www.zhihu.com/question/339723385
    - notebooks/transformers/attention.ipynb
    """
    d_scale = query.size(-1)
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_scale)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -1e9)
    scores = F.softmax(logits, dim=-1)
    output = torch.matmul(scores, value)
    return output, scores

# 存放注意力算法
attentions = {
    'scaled_mul_self_attention': scaled_mul_self_attention
    }


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, 
                hidden_size:int,
                num_attention_heads: int,
                attention_probs_dropout_prob: float,
                return_attention_scores: bool,
                attention_scale: bool= True,
                bias:bool =True,
                **kwargs):

        """多头注意力机制
        参数:
        - hidden_size: 隐层维度
        - num_attention_heads: 注意力头的数量
        - attention_probs_dropout_prob: 注意力机制后dropout的概率
        - return_attention_scores: 是否返回注意力得分
        - attention_scale: 是否对注意力值缩放,默认True
        - bias: 

        参考: 
        - https://github.com/mmmwhy/pure_attention/blob/v0.0.22/pure_attention/backbone_bert/bert_layer.py
        - https://github.com/Tongjilibo/bert4torch/blob/a0db5a59a1ec2ec4820c4d055ceef463ce4e5d28/bert4torch/layers.py#L1
        """
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "隐层维度不能被多头注意力头数整除"
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads 
        self.attention_head_size  = int(hidden_size / num_attention_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.a_bias, self.p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')

        if self.p_bias == 'typical_relative':
            pass

    def forward(self, query, key, value, attention_mask=None, head_mask=None):
        mixed_query_layer = self.q(query)
        mixed_key_layer = self.k(key)
        mixed_value_layer = self.v(value)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores += attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if head_mask is not None:
            attention_probs += head_mask
        context_layer = torch.matmul(attention_scores, value_layer)
        # view操作需要内存连续, permute和transpose之后张量在内存中不再连续所以要加上contiguous
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_size = context_layer.size()[:-2] + (self.hidden_size)
        context_layer = context_layer.view(*new_context_layer_size)
        outputs = (context_layer, attention_scores) if self.return_attention_scores else (context_layer,)
        return outputs

    def transpose_for_scores(self, x):
        new_x_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_size)
        x = x.permute(0, 2, 1, 3)
        return x         

