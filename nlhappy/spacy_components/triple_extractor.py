import shutil
from ..models import BertGlobalPointer, BertGPLinker
from spacy.lang.zh import Chinese
import torch
from spacy.tokens import Doc, Span
from thinc.api import Config
import os
from typing import Union, Optional, List, Tuple


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



class TripleExtractor(object):
    '''spacy组件 三元组抽取
    参数:
    - name: 组件名称
    - ckpt: pl模型保存路径
    - model: pl模型名称
    - device: 设备
    '''
    def __init__(self, nlp, name:str, model:str, ckpt:str, device:str, threshold:float):
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.threshold = threshold
        self.device = torch.device(device)
        self.model_name = model
        self.model_class = models[model]
        self.model = models[model].load_from_checkpoint(ckpt)
        self.model.to(self.device)
        self.model.freeze()
        
        
    def __call__(self, doc: Doc) -> Doc:
        triples = self.model.predict(doc.text, self.device, threshold=self.threshold)
        doc._.triples.extend(triples)
        return doc
    
    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        shutil.copy(self.ckpt, path)
        # 重写NLP配置文件config.cfg 改变pipeline的ckpt路径
        nlp_path = str(path).split('/')[0]
        config_path = os.path.join(nlp_path, 'config.cfg')
        config = Config().from_disk(config_path)
        config['components'][self.pipe_name]['ckpt'] = str(path)
        config.to_disk(config_path)

    def from_disk(self, path:str, exclude):
        self.model = self.model_class.load_from_checkpoint(path)

@Chinese.factory('triple_extractor',assigns=['doc._.triples'],default_config={'model':'bert_gplinker', 'device':'cpu', 'threshold':None})
def make_triple_extractor(nlp, name:str, model:str, ckpt:str, device:str, threshold):
    """三元组抽取组件"""
    if not model in models:
        raise ValueError(f"model must in {models.keys()}")
    else:
        return TripleExtractor(nlp=nlp, name=name, model=model, ckpt=ckpt, device=device, threshold=threshold)



