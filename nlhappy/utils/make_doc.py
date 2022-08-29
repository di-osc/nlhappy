from __future__ import annotations
from spacy.tokens import Token, Span, Doc, DocBin
from typing import Dict, List, Tuple
import srsly
from spacy.language import Language
from tqdm import tqdm 
import random
from nlhappy import chinese
import nlhappy


def get_events(doc: Doc) -> List[Event]:
    events=[]
    for event in doc._.event_data:
        label = event['label']
        roles = event['roles']  
        roles = {k:[ doc.char_span(s[0], s[1]) for s in v ] for k, v in roles.items() if len(v)>0}
        if len(roles)>0:
            events.append(Event(label=label, roles=roles))
    return events


def get_relations(doc: Doc) -> List[Relation]:
    """get the relations in the docs"""
    rels = []
    for rel in doc._.rel_data:
        sub = doc.char_span(rel['sub'][0], rel['sub'][1])
        objs = [doc.char_span(obj[0],obj[1]) for obj in rel['objs']]
        if sub is not None and len(objs) > 0 and None not in objs:
            label = rel['label']
            rels.append(Relation(label=label, sub=sub, objs=objs))
    return rels 


Doc.set_extension('rels', getter=get_relations)
Doc.set_extension('events', getter=get_events)
Span.set_extension('norm_name', default='')
Doc.set_extension('label', default='')

# 数据
Doc.set_extension('rel_data', default=[])
Doc.set_extension('event_data',default=[])


class Event():
    def __init__(self,
                 label: str,
                 roles: Dict[str, List[Span]]) -> None:
        """事件类 
        roles:
            label (str): 事件类型
            roles (Dict[str, List[Span]]): 事件中的角色包括触发词
            
        """
        assert isinstance(label, str) and label != '', 'label must be a string, and not empty, but got {}'.format(label)
        assert isinstance(roles, dict) and len(roles) > 0, 'roles must be a dict, and not empty, but got {}'.format(roles)
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
        if self.label != event.label:
            return False
        if self.roles.keys() != event.roles.keys():
            return False
        for arg in self.roles.keys():
            if self.roles[arg] != event.roles[arg]:
                return False
        return True
    
    def __hash__(self) -> int:
        return hash(self.__class__)
    
    
class EventData(dict):
    def __init__(self,
                 label: str='',
                 roles: dict = {}):
        super().__init__(label=label, roles=roles)
        self.label = label
        self.roles = roles
        
    def __setitem__(self, key, value):
        assert key in ['label', 'roles'], 'key must be sub, label or objs'
        if key == 'label':
            assert type(value) == str, 'label must be str type'
        if key == 'roles':
            assert type(key) == dict, 'roles must be dict type'
            for role in value:
                assert type(value[role]) == list
                for offset in value[role]:
                    assert offset[0]<offset[1], 'offset start must less than end'
        super().__setitem__(key, value)
        
    
class Relation():
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
        super().__init__()
        assert type(label) == str, 'label must be a string, but got {}'.format(type(label))
        assert type(sub) == Span, 'sub must be a Span, but got {}'.format(type(sub))
        assert type(objs) == list, 'objs must be a list of Span, but got {}'.format(type(objs))
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
        return f'Relation({self.sub},{self.label},{self.objs})'
    
    def __eq__(self, rel) -> bool:
        if rel.label != self.label:
            return False
        if self.sub.start_char != rel.sub.start_char or self.sub.end_char != rel.sub.end_char:
            return False
        if len(self.objs) != len(rel.objs):
            return False 
        for i in range(len(self.objs)):
            if self.objs[i].start_char != rel.objs[i].start_char or self.objs[i].end_char != rel.objs[i].end_char:
                return False
        return True
    
    def __hash__(self) -> int:
        return hash(self.__class__)


class RelationData(dict):
    """存放关系offset数据
    
    args:
        sub(tuple): offset of subject
        label(str): label of relation
        objs(List): offsets of objs
    """
        
    def __init__(self, sub=(), label='', objs=[]):
        super().__init__(sub=sub,label=label,objs=objs)
        self.sub= sub
        self.label = label
        self.objs = objs

    def __setitem__(self, key, value):
        assert key in ['sub', 'label', 'objs'], 'key must be sub, label or objs'
        if key == 'sub':
            assert type(value)== tuple, 'sub must be type tuple'
            assert value[0] < value[1], 'offset start must less than end'
        elif key == 'label':
            assert type(value) == str, 'label must be str type'
        elif key == "objs":
            assert type(value) == list
            for obj in value:
                assert obj[0] < obj[1], 'offset start must less than end'     
        super().__setitem__(key, value)
        
        
def make_docs_from_doccano_jsonl(file_path: str, 
                                 nlp: Language = None, 
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
    if not nlp:
        nlp = nlhappy.nlp()
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
                            doc._.rel_data.append(RelationData(sub=(sub.start_char, sub.end_char),
                                                          label=label,
                                                          objs=[(obj.start_char, obj.end_char) for obj in objs]))
            
            docs.append(doc)
        
    return docs


def extend_inverse_relations(docs: List[Doc], 
                             inverse_relations: Dict[str, List[str]]) -> List[Doc]:
    """extend the relations in the docs with inverse relations

    Args:
        docs (List[Doc]): the list of docs
        inverse_relations (Dict[str, List[str]]): the inverse relations, key is the relation label, value is the list of inverse relation labels

    Returns:
        List[Doc]: the list of docs with extended relations
    """
    for doc in tqdm(docs):
        for rel in doc._.rel_data:
            if rel.label in inverse_relations:
                for obj in rel.objs:
                    doc._.rel_data.append(RelationData(label=random.sample(inverse_relations[rel.label], k=1)[0], 
                                                  sub=(obj[0], obj[1]), 
                                                  objs=[(rel.sub[0], rel.sub[1])]))
    return docs



def get_docs_from_docbin(db_path: str,
                         nlp: Language = None):
    """get all docs for docbin 

    Args:
        db_path (str): the path to docbin
        nlp (Language, optional): nlp that uses to get vocab. Defaults to None.
    """
    if not nlp:
        nlp = nlhappy.nlp()
    db = DocBin().from_disk(db_path)
    docs = list(db.get_docs(nlp.vocab))
    return docs
