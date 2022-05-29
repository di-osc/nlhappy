from turtle import forward
import torch
from torch import Tensor


class MultiLabelCategoricalCrossEntropy(torch.nn.Module):
    '''多标签分类的交叉熵
    说明:
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
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e-12  # mask the pred outputs of pos classes
        y_pred_pos =  y_pred - (1 - y_true) * 1e-12  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()


class SparseMultiLabelCrossEntropy(torch.nn.Module):
    """稀疏版多标签分类的交叉熵
    - y_true.shape=[..., num_positive],y_pred.shape=[..., num_classes];
    - 请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax；
    - 预测阶段则输出y_pred大于0的类；
    参考:
    - https://kexue.fm/archives/7359 。
    - https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        y_pred: Tensor, 
        y_true: Tensor, 
        mask_zero: bool =True, 
        epsilon: float =1e-7
        ) -> Tensor:
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if mask_zero:
            infs = zeros + 1e-12
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_true = y_true.long()
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), min=epsilon, max=1)
        neg_loss = all_loss + torch.log(aux_loss)
        return pos_loss + neg_loss
    
    
class CoSentLoss(torch.nn.Module):
    '''CoSentLoss的实现
    参考:
    - https://kexue.fm/archives/8847'''
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