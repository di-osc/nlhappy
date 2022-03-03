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
        if len(self.triple) == 5:
            return self.triple[0:2]
        elif len(self.triple) == 3 :
            return self.triple[0]
    @property
    def predicate(self):
        if len(self.triple) == 5:
            return self.triple[2]
        elif len(self.triple) == 3:
            return self.triple[1]
    
    @property
    def object(self):
        if len(self.triple) == 5:
            return self.triple[3:5]
        elif len(self.triple) == 3:
            return self.triple[2]


class SO:
    """subject和object类"""
    def __init__(
        self, 
        doc: Doc, 
        start: int, 
        end: int
    ) -> None:
        super().__init__()
        self.doc = doc
        self.span = doc[start: end]
        self.text = self.span.text
    
    @property
    def ents(self):
        return self.span.ents

    @property   
    def type_(self):
        """查找subject或者object的类型"""
        if len(self.ents) == 1 and len(self.ents[0]) == len(self.span):
            return self.ents[0].label_
        elif len(self.doc.spans) > 0:
            for span in self.doc.spans['all']:
                if span.start_char == self.span.start_char and span.end_char == self.span.end_char:
                    return span.label_
        else: return None


    @property
    def ents(self):
        return self.span.ents

    @property
    def sent(self):
        return self.span.sent

    @property
    def sents(self):
        return self.span.sents

    def __repr__(self) -> str:
        return f'{self.span}'

    def __hash__(self):
        return self.span.__hash__()

    def __eq__(self, so):
        return self.span.start_char == so.span.start_char and self.span.end_char == so.span.end_char
    
    def __len__(self):
        return len(self.span)

    def __getitem__(self, index):
        return self.span[index]



class SPO(tuple):
    """主谓宾三元组"""
    
    def __init__(self, spo: Tuple[SO, str]):
        super().__init__()
        self.spo = spo
        
    def __hash__(self):
        return self.spo.__hash__()

    def __eq__(self, spo):
        return self.spo == spo.spo

    def __repr__(self) -> str:
        return str(self.spo)

    def __len__(self) -> int:
        return len(self.spo)

    def __getitem__(self, index):
        return self.spo[index]

    @property
    def subject(self):
        return self.spo[0]

    @property
    def predicate(self):
        return self.spo[1]
    
    @property
    def object(self):
        return self.spo[2]