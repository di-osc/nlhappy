from spacy.tokens import Token, Span, Doc
from typing import Dict, List 




Doc.set_extension('events', default=[])


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
        left_idx = min([span.start for val in self.roles.values() for span in val])
        right_idx = max([span.end for val in self.roles.values() for span in val])
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


    

    