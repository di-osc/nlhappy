from torchmetrics import Metric 
from typing import List, Set, Union, Optional
import torch
from collections import namedtuple

Role = namedtuple('Role', ['start', 'end', 'label'])
Clique = namedtuple('Clique', ['hosts'])

class Event:
    def __init__(self, 
                 label: Union[str, int, torch.Tensor], 
                 roles: Set[List[Role]]):
        super().__init__()
        self.label = label
        self.roles = roles

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, __o: object) -> bool:
        return __o.label == self.label and __o.roles == self.roles

    def __repr__(self) -> str:
        return f'Event: <{self.label}>'
    
    


class EventF1(Metric):
    """事件抽取评价指标
    """

    full_state_update: Optional[bool] = False

    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: List[Set[Event]], true: List[Set[Event]]) -> None:
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
        f1 =  2 * self.correct / (self.all_pred + self.all_true + 1e-12)
        return f1
    
    
class CliqueF1(Metric):
    """完全图评价指标,一个完全图每个节点代表着所属一个事件的论元角色
    """
    full_state_update: Optional[bool] = False

    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_pred', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('all_true', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred: List[Set[Clique]], true: List[Set[Clique]]) -> None:
        """
        参数:
        - pred: batch_size大小的列表, 列表内为每个样本的预测的完全图的集合(set)
        - true: 同pred
        """
        assert len(pred) == len(true), f'pred : {pred}, true : {true}'
        for p, t in zip(pred, true):
            self.correct += len(p & t)
            self.all_pred += len(p)
            self.all_true += len(t)
    
    def compute(self):
        # 下面这个公式可以通过化简得到
        f1 =  2 * self.correct / (self.all_pred + self.all_true + 1e-12)
        return f1
    