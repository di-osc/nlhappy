import torch
import torch.nn as nn


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