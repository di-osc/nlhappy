from pydantic import BaseModel
from typing import Tuple, Optional
from torchmetrics import Metric
import torch
from typing import List, Tuple, Set


class SO(BaseModel):
    """主体或者客体"""
    offset : Tuple
    label: Optional[str] = None
    text: Optional[str] = None
    
    def __hash__(self):
        return hash(self.label)
    
    def __eq__(self, other: "SO") -> bool:
        return self.offset == other.offset and self.label == other.label
    

class Relation(BaseModel):
    sub: SO
    obj: SO
    predicate: str
    
    def __hash__(self):
        return hash(self.predicate)
    
    def __eq__(self, other: "Relation") -> bool:
        return self.sub == other.sub and self.predicate == other.predicate and self.obj == other.obj
    
    def __str__(self) -> str:
        if self.sub.text is None or self.obj.text is None:
            return f"{self.sub.offset}-{self.predicate}-{self.obj.offset}"
        else: 
            return f"{self.sub.text}-{self.predicate}-{self.obj.text}"
        
    def __repr__(self) -> str:
        if self.sub.text is None or self.obj.text is None:
            return f"{self.sub.offset}-{self.predicate}-{self.obj.offset}"
        else: 
            return f"{self.sub.text}-{self.predicate}-{self.obj.text}"
    
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