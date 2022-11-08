import torch
from torch import nn
from torch.nn import Module
from ..embedding import SinusoidalPositionEmbedding, RoPEPositionEncoding


class GlobalPointer(Module):
    """全局指针模块,将序列的每个(start, end)作为整体来进行判断
    
    更改
    - 原版模型用矩阵点乘来得到span的表征,这里增加了叉乘,相加和拼接的方法,用来做对比实验
    """
    def __init__(self,
                 input_size: int, 
                 hidden_size: int,
                 output_size: int,  
                 add_rope: bool =True,
                 tril_mask: bool = True,
                 span_get_type: str = 'dot'):
        """

        Args:
            input_size (int): 输入维度大小一般为bert输出的hidden size
            hidden_size (int): 内部隐层的维度大小
            output_size (int): 一般为分类的span类型
            add_rope (bool, optional): 是否添加RoPE. Defaults to True.
            tril_mask (bool, optional): 是否遮掩下三角. Defaults to True.
            span_get_type (str, optional): span表征的获得方式.可以为'dot' 'element-product' 'concat' 'element-add',默认'dot',实验dot也是最好.
        """
        super(GlobalPointer, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.add_rope = add_rope
        self.tril_mask = tril_mask
        self.span_get_type = span_get_type
        if self.span_get_type == 'dot':
        # 加了relu激活函数在hidden_size很小(32)的情况下,损失不下降,所以去掉了ReLU激活函数
        # self.q = nn.Sequential(nn.Linear(input_size, hidden_size * output_size), nn.ReLU())
        # self.k = nn.Sequential(nn.Linear(input_size, hidden_size * output_size), nn.ReLU())
            self.start = nn.Linear(input_size, hidden_size * output_size)
            self.end = nn.Linear(input_size, hidden_size * output_size)
        if self.span_get_type == 'element-product' or self.span_get_type == 'element-add':
            self.start = nn.Linear(input_size, hidden_size)
            self.end = nn.Linear(input_size, hidden_size) 
            self.span = nn.Linear(hidden_size, output_size)
        if self.span_get_type == 'concat':
            self.start = nn.Linear(input_size, hidden_size)
            self.end = nn.Linear(input_size, hidden_size) 
            self.span =  nn.Linear(hidden_size*2, output_size)


    def forward(self, inputs, mask=None):
        if self.span_get_type == 'dot':
            batch_size, seq_length, input_size = inputs.shape
            start_logits = self.start(inputs).reshape(batch_size, seq_length, self.output_size, self.hidden_size)
            end_logits = self.end(inputs).reshape(batch_size, seq_length, self.output_size, self.hidden_size)
            # 分出qw和kw
            # RoPE编码
            if self.add_rope:
                pos = SinusoidalPositionEmbedding(self.hidden_size, 'zero')(inputs)
                cos_pos = pos[..., None, 1::2].repeat(1, 1, 1, 2)
                sin_pos = pos[..., None, ::2].repeat(1, 1, 1, 2)
                start_logits2 = torch.stack([-start_logits[..., 1::2], start_logits[..., ::2]], 4)
                start_logits2 = torch.reshape(start_logits2, start_logits.shape)
                start_logits = start_logits * cos_pos + start_logits2 * sin_pos
                end_logits2 = torch.stack([-end_logits[..., 1::2], end_logits[..., ::2]], 4)
                end_logits2 = torch.reshape(end_logits2, end_logits.shape)
                end_logits = end_logits * cos_pos + end_logits2 * sin_pos
            # 计算内积
            logits = torch.einsum('bmhd , bnhd -> bhmn', start_logits, end_logits)
            logits = logits / self.hidden_size ** 0.5
        elif self.span_get_type == 'element-product' or self.span_get_type == 'element-add':
            start_logits = self.start(inputs)
            end_logits = self.end(inputs)
            if self.add_rope:
                pos = SinusoidalPositionEmbedding(self.hidden_size, 'zero')(inputs)
                cos_pos = pos[..., 1::2].repeat(1, 1, 2)
                sin_pos = pos[..., ::2].repeat(1, 1, 2)
                start2 = torch.stack([-start_logits[..., 1::2], start_logits[..., ::2]], 3)
                start2 = torch.reshape(start2, start_logits.shape)
                start_logits = start_logits * cos_pos + start2 * sin_pos
                end2 = torch.stack([-end_logits[..., 1::2], end_logits[..., ::2]], 3)
                end2 = torch.reshape(end2, end_logits.shape)
                end_logits = end_logits * cos_pos + end2 * sin_pos
            start_logits = start_logits.unsqueeze(1)
            end_logits = end_logits.unsqueeze(2)
            if self.span_get_type == 'element-product':
                logits = self.span(start_logits * end_logits).permute(0,3,2,1) # batch output seq seq
            elif self.span_get_type == 'element-add':
                logits = self.span(start_logits + end_logits).permute(0,3,2,1)
                
        elif self.span_get_type == 'concat':
            batch_size, seq_len, input_size = inputs.size()
            # head: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,1,1, 2,2,2, 3,3,3
            start_logits = self.start(inputs)
            end_logits = self.end(inputs)
            if self.add_rope:
                pos = SinusoidalPositionEmbedding(self.hidden_size, 'zero')(inputs)
                cos_pos = pos[..., 1::2].repeat(1, 1, 2)
                sin_pos = pos[..., ::2].repeat(1, 1, 2)
                start2 = torch.stack([-start_logits[..., 1::2], start_logits[..., ::2]], 3)
                start2 = torch.reshape(start2, start_logits.shape)
                start_logits = start_logits * cos_pos + start2 * sin_pos
                end2 = torch.stack([-end_logits[..., 1::2], end_logits[..., ::2]], 3)
                end2 = torch.reshape(end2, end_logits.shape)
                end_logits = end_logits * cos_pos + end2 * sin_pos
            start_logits = start_logits.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.hidden_size).reshape(batch_size, seq_len*seq_len, self.hidden_size)
            # end: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,2,3, 1,2,3, 1,2,3
            end_logits = end_logits.repeat(1, seq_len, 1)
            pairs = torch.cat([start_logits, end_logits], dim=-1)
            logits = self.span(pairs).reshape(batch_size, self.output_size, seq_len, seq_len)
            
        # padding mask
        if mask is not None:
            batch_size = inputs.size()[0]
            seq_len = inputs.size()[1]
            pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.output_size, seq_len, seq_len)
            logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1) 
            logits = logits - mask * 1e12

        # scale返回
        return logits



class EfficientGlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    https://kexue.fm/archives/8877
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 output_size: int,
                 add_rope: bool =True,
                 tril_mask: bool =True,
                 use_bias: bool = True,
                 max_length: int = 512):
        super(EfficientGlobalPointer, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.add_rope = add_rope
        self.tril_mask = tril_mask
        self.input_size = input_size
        self.linear_1 = nn.Linear(input_size, hidden_size * 2, bias=use_bias)
        self.linear_2 = nn.Linear(hidden_size * 2, output_size * 2, bias=use_bias)
        if self.add_rope:
            self.pe = RoPEPositionEncoding(max_position=max_length, embedding_size=hidden_size)

    def forward(self, inputs, mask=None):
        inputs = self.linear_1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.add_rope:
            qw = self.pe(qw)
            kw = self.pe(kw)
        logits = torch.einsum('bmd , bnd -> bmn', qw, kw) / self.hidden_size ** 0.5
        # bias = self.linear_2(inputs)
        # bias = torch.stack(torch.chunk(bias, self.output_size, dim=-1), dim=-2).transpose(1,2) #[btz, heads, seq_len, 2]
        # logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)
        bias = self.linear_2(inputs).transpose(1, 2) / 2  #'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        
        # padding mask
        if mask is not None:
            batch_size = inputs.size()[0]
            seq_len = inputs.size()[1]
            pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.output_size, seq_len, seq_len)
            logits = logits*pad_mask - (1-pad_mask)*1e12

        # tril mask
        if self.tril_mask:
            t_mask = torch.tril(torch.ones_like(logits), -1) 
            logits = logits - t_mask * 1e12

        return logits