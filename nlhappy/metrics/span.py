from torch import Tensor
from torchmetrics import Metric
import torch
from typing import List, Set, Optional


class SpanF1(Metric):

    full_state_update: Optional[bool] = False

    """计算span矩阵的F1"""
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