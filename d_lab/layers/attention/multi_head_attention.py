import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import math

# 乘性的注意力算法
def scaled_mul_self_attention(query, key, value, mask=None):
    """
    # 为什么scaled(也就是除以sqrt(d_k))？: 
    - softmax梯度在数量级别较大的时候会造成梯度消失
    - 假设q, k的各个分量是互相独立的随机分布,均值为0,方差为1,那么q点积k的均值为0,方差为d_k,
    - 所以除以sqrt(d_k)为了缓解softmax的梯度消失
    - https://www.zhihu.com/question/339723385
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


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(
        self, 
        n_head:int,
        d_in:int,
        attention_type: str = 'scaled_mul_self_attention'
        ):
        """
        - n_head: 头数
        - d_in: 输入张量的特征维度
        - p_drop: dropout概率
        """
        super().__init__()
        assert d_in % n_head == 0, "d_in must be divisible by n_head"
        self.d_in = d_in
        self.n_head = n_head
        self.attention = attentions[attention_type]
        self.linears = nn.ModuleList([
            nn.Linear(d_in, d_in) for _ in range(4)
        ])

    def forward(self, q, k, v, mask=None, return_attention=False):
        q_view, k_view, v_view = [model(x).view(q.size(0), -1, self.n_head, self.d_in//self.n_head).transpose(1,2) for model, x in zip(self.linears, (q, k, v))]
        q_out, q_attn = self.attention(q_view, k_view, v_view, mask)
        q_out = q_out.transpose(1,2).contiguous().view(q.size(0), -1, self.d_in)
        q_attn = q_attn.transpose(1,2).contiguous().view(q.size(0), -1, self.d_in)
        q_out = self.linears[-1](q_out)
        if return_attention:
            return q_out, q_attn
        else:
            return q_out

if __name__ == "__main__":
    input = torch.randn(2, 10, 20)
    model = MultiHeadAttention(n_head=2, d_in=20)
    outputs = model(input, input, input, return_attention=True)
    print(outputs[1].shape)
