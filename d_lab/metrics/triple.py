from torchmetrics import Metric
import torch
from typing import Tuple, Set, List
from ..utils.data import Triple

class TripleF1(Metric):
    """三元组抽取的F1
    说明:
    - 输入为(subject_start, subject_end, predicate, object_start, object_end)
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: List[Set[Triple]], true: List[Set[Triple]]) -> None:
        """
        参数:
        - pred: batch_size大小的列表, 列表内为每个样本的预测的三元组的集合(set)
        - true: 同pred
        """
        assert len(pred) == len(true)
        for p, t in zip(pred, true):
            self.correct += len(p & t)
            self.all_pred += len(p)
            self.all_true += len(t)
    
    def compute(self):
        # 下面这个公式可以通过化简得到
        return 2 * self.correct / (self.all_pred + self.all_true + 1e-10)