import torch
from torch import Tensor
import numpy as np


class MultiLabelCategoricalCrossEntropy(torch.nn.Module):
    '''用于多标签分类的交叉熵
    说明:
    - 阈值应为0
    - y_true和y_pred的shape一致,y_true的元素非0即1,1表示对应的类为目标类,0表示对应的类为非目标类;
    - 请保证y_pred的值域是全体实数;换言之一般情况下y_pred不用加激活函数,尤其是不能加sigmoid或者softmax;
    - 预测阶段则输出y_pred大于0的类;
    参考:
    - https://kexue.fm/archives/7359 
    - https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L250
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
            y_pred : [batch_size * num_classes, seq_len * seq_len]
            y_true : [batch_size * num_classes, seq_len * seq_len]
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos =  y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
        
        zeros = torch.zeros_like(y_pred[..., :1])
        
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = neg_loss + pos_loss
        if len(loss.shape) == 3:
            loss = loss.sum(dim=1).mean()
        else:
            loss = loss.mean()
        return loss

class SparseMultiLabelCrossEntropy(torch.nn.Module):
    """token-pair稀疏版多标签分类的交叉熵, y_true只传正例
    - y_true.shape=[batch, class, max_instance],
    - y_pred.shape=[batch, class,];
    - 请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax；
    - 预测阶段则输出y_pred大于0的类；
    参考:
    - https://kexue.fm/archives/7359 。公式5
    - https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
    """
    def __init__(self, mask_zero: bool = True, eposilon: float = 1e-7, inf=1e7):
        super().__init__()
        self.eposilon = eposilon
        self.mask_zero = mask_zero
        self.inf = inf
        
    def forward(self, y_pred: Tensor, y_true: Tensor):
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1) # 最后添加一个0 
        if self.mask_zero:
            infs = zeros + self.inf
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
            
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        # 计算负类对应损失
        all_loss = torch.logsumexp(y_pred, dim=-1) # 公式里面的a
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss # b - a
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=self.eposilon, max=1) # 1-exp(b-a)
        neg_loss = all_loss + torch.log(aux_loss) # a + log[1-exp(b-a)]
        loss = pos_loss + neg_loss
        return loss.sum(dim=1).mean()
    
        # pos_loss = torch.logsumexp(-1 * torch.cat([torch.gather(y_pred, index=y_true, dim=-1), torch.zeros_like(y_pred[..., :1])], dim=-1), dim=-1)
        # a = torch.logsumexp(torch.cat([y_pred, torch.zeros_like(y_pred[..., :0])], dim=-1), dim=-1)
        # b = torch.logsumexp(torch.gather(y_pred, index=y_true, dim=-1), dim=-1)
        # neg_loss = a + torch.log(torch.clamp(1 - torch.exp(b - a), min=self.eposilon, max=1))
        # loss = (pos_loss + neg_loss).sum(dim=1).mean()
        # return loss
    
    
class CoSentLoss(torch.nn.Module):
    '''CoSentLoss的实现
    参考:
    - https://kexue.fm/archives/8847
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds, targs, device):
        '''
        参数:
        - preds: 余弦相似度 [0.1, 0.2, 0.3,...]
        - targs: 标签 [0,1,1,0, ...]
        - device: 运行的设备
        返回:
        - loss: 余弦相似度与标签的交叉熵
        '''
        # 20为公式里面的lambda 这里取20
        preds = preds * 20
        # 利用广播机制, 所有位置, 两两差值
        preds = preds[:, None] - preds[None, :]
        # 
        targs = targs[:, None] < targs[None, :]
        targs = targs.float()
        # 这里之所以要这么减，是因为公式中所有的正样本对的余弦值减去负样本对的余弦值才计算损失，把不是这些部分通过exp(-inf)忽略掉
        preds = preds - (1 - targs) * 1e12
        preds = preds.view(-1)
        preds = torch.cat((torch.tensor([0], dtype=torch.float, device=device), preds), dim=0)
        return torch.logsumexp(preds, dim=0)