import torch
import torch.nn as nn
from ..embedding import SinusoidalPositionEmbedding


class Biaffine(nn.Module):
    def __init__(self, input_size, output_size=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((output_size, input_size + int(bias_x), input_size + int(bias_y)))
        nn.init.xavier_normal_(weight)                                          #将权重变为正态分布
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"input_size={self.input_size}, output_size={self.output_size}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1) #加偏执项
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, output_size, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)  #矩阵点积
        # remove dim 1 if output_size == 1
        s = s.permute(0, 2, 3, 1)

        return s  
    
    
class BiaffineSpanClassifier(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_size: int,
                 add_rope: bool = True,
                 tril_mask: bool = True):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.tril_mask = tril_mask
        self.add_rope = add_rope
        self.start_repr = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.end_repr = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        weight = torch.zeros((output_size, hidden_size, hidden_size))
        nn.init.xavier_normal_(weight)                                          #将权重变为正态分布
        self.weight = nn.Parameter(weight, requires_grad=True)


    def forward(self, x, mask=None):
        """将序列的表示(batch,seq,hidden),转换为token pair的表示

        Args:
            x : [batch_size, seq_len, hidden_size]
            mask : [batch_size, seq_len]

        Returns:
            logits: [batch_size, output_size, seq_len, seq_len]
        """
        start_logits = self.start_repr(x)
        end_logits = self.end_repr(x)
        if self.add_rope:
            pos = SinusoidalPositionEmbedding(self.hidden_size, 'zero')(x)
            cos_pos = pos[..., 1::2].repeat(1, 1, 2)
            sin_pos = pos[..., ::2].repeat(1, 1, 2)
            start2 = torch.stack([-start_logits[..., 1::2], start_logits[..., ::2]], 3)
            start2 = torch.reshape(start2, start_logits.shape)
            start_logits = start_logits * cos_pos + start2 * sin_pos
            end2 = torch.stack([-end_logits[..., 1::2], end_logits[..., ::2]], 3)
            end2 = torch.reshape(end2, end_logits.shape)
            end_logits = end_logits * cos_pos + end2 * sin_pos

        # [batch_size, output_size, seq_len, seq_len]
        span_logits = torch.einsum('bxi,oij,byj->boxy', start_logits, self.weight, end_logits)  #矩阵点积            
        
        
        if mask is not None:
            batch_size = x.size()[0]
            seq_len = x.size()[1]
            pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.output_size, seq_len, seq_len)
            span_logits = span_logits*pad_mask - (1-pad_mask) * 1e12
        
        if self.tril_mask:
            mask_tril = torch.tril(torch.ones_like(span_logits), diagonal=-1)
            span_logits = span_logits - mask_tril * 1e12
            

        return span_logits  