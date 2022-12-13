from torch import Tensor
from torchmetrics import Metric
import torch
from typing import List, Set, Optional


class SpanF1(Metric):
    """计算span矩阵的F1"""

    full_state_update: Optional[bool] = False

    def __init__(self):
        super().__init__()

        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: Tensor, true: Tensor):
        self.correct += torch.sum(pred[true==1])
        self.all_pred += torch.sum(pred == 1)
        self.all_true += torch.sum(true == 1)

    def compute(self):
        return 2 * self.correct / (self.all_pred + self.all_true)
    
    
class SpanTokenF1(Metric):
    """以预测的span中的token正确为评判指标
    - 输入的形式为[0,0,0,1,1,1]
    """
    
    full_state_update: Optional[bool] = False
    
    def __init__(self):
        super().__init__()

        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: Tensor, true: Tensor):
        self.correct += torch.sum(pred[true==1])
        self.all_pred += torch.sum(pred == 1)
        self.all_true += torch.sum(true == 1)

    def compute(self):
        return 2 * self.correct / (self.all_pred + self.all_true)
    
    
class SpanIndexF1(Metric):
    """以预测的下标正确为评判指标
    - 输入的形式为[{0,1,2}, {0,1,2}]
    """
    
    full_state_update: Optional[bool] = False
    
    def __init__(self):
        super().__init__()

        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: List[Set[int]], true: List[Set[int]]):
        for p, t in zip(pred, true):
            if len(t) == 0 and len(p) == 0: # 如果标签和预测的下标都为0,则测试F1为1
                self.correct += 1
                self.all_pred += 1
                self.all_true += 1
            else:
                self.correct += len(p & t)
                self.all_pred += len(p)
                self.all_true += len(t)

    def compute(self):
        return 2 * self.correct / (self.all_pred + self.all_true)