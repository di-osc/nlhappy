import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, 
        hidden_size, 
        eps=1e-12, 
        conditional=False
        ):
        """layernorm 层，这里自行实现，目的是为了兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务
           条件layernorm来自于苏剑林的想法，详情：https://spaces.ac.cn/archives/7124
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.conditional = conditional
        if conditional:
            # 条件layernorm, 用于条件文本生成,
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.dense1.gamma.data.uniform_(0, 0)
            self.dense2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.dense2.gamma.data.uniform_(0, 0)

    def forward(self, x):
        if self.conditional:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.eps)
            return (self.weight + self.dense1(cond)) * x + (self.bias + self.dense2(cond))
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight * x + self.bias