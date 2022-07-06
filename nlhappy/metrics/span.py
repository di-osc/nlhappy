from torch import Tensor
from torchmetrics import Metric
import torch


class SpanF1(Metric):
    """计算span矩阵的F1"""
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: Tensor, true: Tensor):
        self.correct += torch.sum(pred[true==1])
        self.all_pred += torch.sum(pred != 0)
        self.all_true += torch.sum(true != 0)

    def compute(self):
        return 2 * self.correct / (self.all_pred + self.all_true + 1e-5)

        






