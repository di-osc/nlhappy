from torchmetrics import Metric
import torch
from typing import Tuple, Set, List

class Triple(tuple):
    """存放三元组的五元形式(sub_start, sub_end, predicate, obj_start, obj_end), 这样改写方便求交集"""
    
    def __init__(self, triple: Tuple[int, str]):
        super().__init__()
        self.triple = triple
        
    def __hash__(self):
        return self.triple.__hash__()

    def __eq__(self, triple):
        return self.triple == triple.triple

    def __repr__(self) -> str:
        return str(self.triple)

    def __len__(self) -> int:
        return len(self.triple)

    def __getitem__(self, index):
        return self.triple[index]

    @property
    def subject(self):
        """返回subject的下标, 左闭右开"""
        return (self.triple[0], self.triple[1])
       
    @property
    def predicate(self):
        return self.triple[2]
        
    
    @property
    def object(self):
        """返回object的下标, 左闭右开"""
        return (self.triple[3], self.triple[4])

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