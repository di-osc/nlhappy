from spacy.tokens import Token, Span, Doc
from typing import Dict, List 


Doc.set_extension('events', default=[])
Doc.set_extension('relations', default=[])


class Event:
    def __init__(self,
                 label: str,
                 roles: Dict[str, List[Span]]) -> None:
        """事件类 
        roles:
            label (str): 事件类型
            roles (Dict[str, List[Span]]): 事件中的角色包括触发词
            
        """
        
        self.label = label
        self.roles = roles
        
        
    @property
    def sents(self):
        left_idx = min([span.start_char for val in self.roles.values() for span in val])
        right_idx = max([span.end_char for val in self.roles.values() for span in val])
        # return self.doc[left_idx:right_idx].sents  ##################### 这个有个bug 暂时不能这样返回
        sents = list(self.doc[left_idx:right_idx].sents)
        last = sents.pop()
        sents.append(last.sent)
        return sents

        
    @property
    def doc(self) -> Doc:
        for spans in self.roles.values():
            for span in spans:
                return span.doc
        

    def __repr__(self) -> str:
        return f'Event({self.label})'
        
        
    def __eq__(self, event) -> bool:
        if not isinstance(event, Event):
            return False
        if self.label != event.label:
            return False
        if self.roles.keys() != event.roles.keys():
            return False
        for arg in self.roles.keys():
            if self.roles[arg] != event.roles[arg]:
                return False
        return True
    
    
    
class Relation:
    """used to store the relation in the doc
    """
    def __init__(self,
                 label: str,
                 sub: Span,
                 objs: List[Span]) -> None:
        """
        Args:
            label (str): the relation label
            sub (Span): the subject of the relation
            obj (Span): the object of the relation
        """
        self.label = label
        self.sub = sub
        self.objs = objs
        
        
    @property
    def sents(self) -> List[Span]:
        left_idx = min([self.sub.start_char, min([obj.start_char for obj in self.objs])])
        right_idx = max([self.sub.end_char, max([obj.end_char for obj in self.objs])])
        # return self.doc[left_idx:right_idx].sents
        sents = list(self.doc[left_idx:right_idx].sents)
        last = sents.pop()
        sents.append(last.sent)
        return sents
    
    
    @property
    def doc(self) -> Doc:
        return self.sub.doc
        
        
    def __repr__(self) -> str:
        return f'Relation({self.sub},{self.label})'
    
    
    def __eq__(self, rel) -> bool:
        return self.label == rel.label and self.sub == rel.sub and self.obj == rel.obj
    
    
    
        
        
        


    

    