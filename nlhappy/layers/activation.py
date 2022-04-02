import math
import torch
import torch.nn.functional as F

class GELU(torch.nn.Module):
    """
    高斯误差线性单元,该激活函数在激活中加入了**随机正则**的思想
    参考:
    - https://kexue.fm/archives/7309
    - https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class  GELU_Approximate(torch.nn.Module):
    """ 
    gelu激活函数的近似版本
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SWISH(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



activations = {'gelu':GELU, 'swish':SWISH,'gelu_approximate':GELU_Approximate}

