from spacy.tokens import Token, Span, Doc
from typing import Dict, List 


Doc.set_extension('events', default=[])


class Event:
    def __init__(self,
                 label: str,
                 args: Dict[str, List[Span]]) -> None:
        """事件类 
        Args:
            label (str): 事件类型
            args (Dict[str, List[Span]]): 事件的论元包括触发词
            
        """
        
        self.label = label
        self.args = args
        
        
    @property
    def sents(self):
        left_idx = min([span.start for val in self.args.values() for span in val])
        right_idx = max([span.end for val in self.args.values() for span in val])
        # return self.doc[left_idx:right_idx].sents  ##################### 这个有个bug 暂时不能这样返回
        sents = list(self.doc[left_idx:right_idx].sents)
        last = sents.pop()
        sents.append(last.sent)
        return sents

        
        
    @property
    def doc(self) -> Doc:
        for spans in self.args.values():
            for span in spans:
                return span.doc
        

    def __repr__(self) -> str:
        return f'Event({self.label})'
        
        
    def __eq__(self, event) -> bool:
        if not isinstance(event, Event):
            return False
        if self.label != event.label:
            return False
        if self.args.keys() != event.args.keys():
            return False
        for arg in self.args.keys():
            if self.args[arg] != event.args[arg]:
                return False
        return True
    
    
            
            
def make_event_extractor():
    pass
        