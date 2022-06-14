from spacy.tokens import Token, Span, Doc
from typing import Dict, List 
import srsly
from spacy.language import Language
from tqdm import tqdm 


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
    
    
    
def make_docs_from_doccano_jsonl(file_path: str, 
                                 nlp: Language, 
                                 set_ent: bool = True,
                                 set_span: bool = False,
                                 set_relation: bool = True) -> List[Doc]:
    """make docs and add entities from doccano jsonl file

    Args:
        file_path (str): the path of the doccano jsonl file
        nlp (Language): the spacy language processing pipeline
        set_ent (bool, optional): whether to set the entities in the doc. Defaults to True.
        set_span (bool, optional): whether to set the spans with a 'all' key in the doc. uses for overlap spans. Defaults to False.
        set_relation (bool, optional): whether to set the relations in the doc. Defaults to True.

    Returns:
        List[Doc]: the list of doc with tags
    """
    docs = []
    for d in tqdm(srsly.read_jsonl(file_path)):
        doc = nlp(d['text'])
        if len(d['entities']) > 0:
            ents = {}
            for ent in d['entities']:
                ents[ent['id']] = doc.char_span(ent['start_offset'], ent['end_offset'], label=ent['label'])
            
            if set_ent:
                try:
                    doc.set_ents(list(ents.values()))
                except:
                    pass
            if set_span:
                doc.spans['all'] = list(ents.values())
            if set_relation:
                if len(d['relations']) > 0:
                    rels = {}
                    for rel in d['relations']:
                        if rel['from_id'] not in rels:
                            rels[rel['from_id']] = {}
                        if rel['type'] not in rels[rel['from_id']]:
                            rels[rel['from_id']][rel['type']] = []
                        rels[rel['from_id']][rel['type']].append(rel['to_id'])
                    for sub_id in rels:
                        for rel_type in rels[sub_id]:
                            sub = ents[sub_id]
                            objs = [ents[obj_id] for obj_id in rels[sub_id][rel_type]]
                            label = rel_type
                            doc._.relations.append(Relation(label, sub, objs))
            
            docs.append(doc)
        
    return docs
            
            
           
        
        


    

    