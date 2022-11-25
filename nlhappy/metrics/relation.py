from pydantic import BaseModel
from typing import Tuple, Optional
from torchmetrics import Metric
import torch
from typing import List, Tuple, Set
from ..data.doc import Entity, Relation

    
class RelationF1(Metric):
    """实体和关系联合抽取的F1指标
    """

    full_state_update: Optional[bool] = False

    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: List[Set[Relation]], true: List[Set[Relation]]) -> None:
        """
        参数:
        - pred: batch_size大小的列表, 列表内为每个样本的预测的三元组的集合(set)
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