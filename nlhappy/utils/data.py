import torch
from typing import List, Tuple
from spacy.tokens import Doc, Span

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
        

if __name__ == "__main__":
    t = Triple((1,2,3,'啊',5,6,7))
    t1 = Triple((1,2,3,'的',5,6,7))
    print(t==t1)
    print(t.subject)
    print(t.predicate)
    print(t.object)