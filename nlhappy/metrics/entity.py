from torchmetrics import Metric
import torch
from typing import List, Set, Tuple
from collections import namedtuple

Entity = namedtuple('Entity', ['label', 'indexes'])


class EntityF1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: List[Set[Entity]], true: List[Set[Entity]]) -> None:
        """
        参数:
        - pred: batch_size大小的列表, 列表内为每个样本的预测的实体的下标例如(1,2,3) 非连续的为(1,2,4)
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