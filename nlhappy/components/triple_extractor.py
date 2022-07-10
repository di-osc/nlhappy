import pickle
from ..models import BertGPLinker
from ..utils.make_doc import RelationData
from spacy.lang.zh import Chinese
import torch
from spacy.tokens import Doc, Span
from thinc.api import Config
import os
from typing import Union, Optional, List, Tuple
import logging

log = logging.getLogger(__name__)

class SO:
    """subject和object类"""
    def __init__(
        self, 
        doc: Doc, 
        start: int, 
        end: int
    ) -> None:
        super().__init__()
        self.start_char = start
        self.end_char = end
        self.doc = doc
    
    @property
    def span(self) -> Span:
        return self.doc.char_span(self.start_char, self.end_char)

    @property
    def text(self) -> str:
        return self.span.text
    
    @property   
    def label_(self):
        """查找subject或者object的类型"""
        if len(self.ents) == 1 and len(self.ents[0]) == len(self.span):
            return self.ents[0].label_
        elif len(self.doc.spans) > 0:
            for span in self.span.doc.spans['all']:
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


def get_spoes(doc):
    spoes = []
    for triple in doc._.triples:
        sub = SO(doc, triple[0], triple[1])
        pred = triple[2]
        obj = SO(doc, triple[3], triple[4])
        spoes.append(SPO((sub, pred, obj)))
    return spoes


models = {'bert_gplinker': BertGPLinker}
Doc.set_extension('triples', default=[])
Doc.set_extension('spoes', getter=get_spoes)


class GPLinkerExtractor(object):
    '''spacy组件 三元组抽取
    参数:
    - name: 组件名称
    - ckpt: pl模型保存路径
    - model: pl模型名称
    - device: 设备
    '''
    def __init__(self, 
                 nlp, 
                 name:str, 
                 device:str, 
                 threshold:float,
                 set_rels:bool):
        self.nlp = nlp
        self.pipe_name = name
        self.threshold = threshold
        self.device = device
        self.set_rels = set_rels
        
    def __call__(self, doc: Doc) -> Doc:
        triples = self.model.predict(doc.text, self.device, threshold=self.threshold)
        doc._.triples.extend(triples)
        if self.set_rels:
            for triple in triples:
                sub_offset = (triple[0], triple[1])
                doc._.rel_data.append(RelationData(sub=sub_offset, label=triple[2], objs=[(triple[3], triple[4])]))
        return doc
    
    def init_model(self, model_or_path):
        if isinstance(model_or_path, BertGPLinker):
            self.model = model_or_path
            self.model.freeze()
            self.model.to(self.device)
            
        else:
            self.model= BertGPLinker.load_from_checkpoint(model_or_path)
            self.model.freeze()
            self.model.to(self.device)
    
    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        # path : save_path/information_extractor
        if not os.path.exists(path):
            os.mkdir(path=path)
        model = 'te.pkl'
        model_path = os.path.join(path, model)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
    def from_disk(self, path:str, exclude):
        model_path = os.path.join(path, 'te.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model.freeze()
        try:
            self.model.to(self.device)
        except:
            log.info(f' to device {self.device} failed')

default_config = {'threshold':0.5,
                  'device':'cpu',
                  'set_rels': 'True',
                  'model':'GPLinker'}

@Chinese.factory('triple_extractor',assigns=['doc._.triples', 'doc._.relations'],default_config=default_config)
def make_triple_extractor(nlp, 
                          name:str, 
                          device:str, 
                          threshold:float,
                          set_rels:bool,
                          model:str):
    """三元组抽取组件"""
    if model == 'GPLinker':
        return GPLinkerExtractor(nlp=nlp, 
                                 name=name, 
                                 device=device, 
                                 threshold=threshold,
                                 set_rels= set_rels)



