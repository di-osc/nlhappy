import torch
from torch import nn
from torch.nn import Module
from ..embedding import SinusoidalPositionEmbedding

def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits


class GlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(
        self,
        input_size: int, 
        hidden_size: int,
        output_size: int,  
        RoPE: bool =True,
        tril_mask: bool = True):
        super(GlobalPointer, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.fc = nn.Linear(input_size, hidden_size * output_size * 2)

#     def reset_params(self):
#         nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, inputs, mask=None):
        inputs = self.fc(inputs)
        inputs = torch.split(inputs, self.hidden_size * 2, dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.hidden_size], inputs[..., self.hidden_size:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.hidden_size, 'zero')(inputs)
            cos_pos = pos[..., None, 1::2].repeat(1, 1, 1, 2)
            sin_pos = pos[..., None, ::2].repeat(1, 1, 1, 2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # padding mask
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.output_size, seq_len, seq_len)
    
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1) 
            logits = logits - mask * 1e12

        # scale返回
        return logits / self.hidden_size ** 0.5



class EfficientGlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    https://kexue.fm/archives/8877
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        output_size: int,
        RoPE: bool =True,
        tril_mask: bool =True):
        super(EfficientGlobalPointer, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.input_size = input_size
        self.linear_1 = nn.Linear(input_size, hidden_size * 2, bias=True)
        self.linear_2 = nn.Linear(hidden_size * 2, output_size * 2, bias=True)

    def forward(self, inputs, mask=None):
        inputs = self.linear_1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.hidden_size, 'zero')(inputs)
            cos_pos = pos[...,1::2].repeat(1,1,2)
            sin_pos = pos[...,::2].repeat(1,1,2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd , bnd -> bmn', qw, kw) / self.hidden_size**0.5
        bias = torch.einsum('bnh -> bhn', self.linear_2(inputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        
        # padding mask
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.output_size, seq_len, seq_len)
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # tril mask
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1) 
            logits = logits - mask * 1e12

        return logits