from torch import Tensor
from torchmetrics import Metric
import torch
from typing import List, Set 


class SpanF1(Metric):
    """计算span矩阵的F1"""
    def __init__(self):
        super().__init__()

        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: Tensor, true: Tensor):
        self.correct += torch.sum(pred[true==1])
        self.all_pred += torch.sum(pred != 0)
        self.all_true += torch.sum(true != 0)

    def compute(self):
        return 2 * self.correct / (self.all_pred + self.all_true + 1e-5)


class SpanOffsetF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: List[Set[tuple]], true: List[Set[tuple]]) -> None:
        """
        参数:
        - pred: batch_size大小的列表, 列表内为每个样本的预测的span offset的集合(set)
        - true: 同pred
        """
        assert len(pred) == len(true), f'pred : {pred}, true : {true}'
        for p, t in zip(pred, true):
            self.correct += len(p & t)
            self.all_pred += len(p)
            self.all_true += len(t)
    
    def compute(self):
        # 下面这个公式可以通过化简得到
        return 2 * self.correct / (self.all_pred + self.all_true + 1e-10)