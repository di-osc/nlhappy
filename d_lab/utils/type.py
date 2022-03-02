import torch
from typing import List, Tuple

class Triple(tuple):
    """为了存放关系抽取三元组, 这样改写方便求交集"""
    
    def __init__(self, triple: Tuple[int]):
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