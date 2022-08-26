import pickle
from ..models import GPLinkerForPromptRelationExtraction, GPLinkerForRelationExtraction
from ..utils.make_doc import RelationData
from spacy.lang.zh import Chinese
from spacy.tokens import Doc
import os
from typing import Union, Optional, List, Tuple
import logging

log = logging.getLogger(__name__)

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
                 threshold:float):
        self.nlp = nlp
        self.pipe_name = name
        self.threshold = threshold
        self.device = device
        
    def __call__(self, doc: Doc) -> Doc:
        triples = self.model.predict(doc.text, self.device, threshold=self.threshold)
        for triple in triples:
            sub_offset = (triple[0], triple[1])
            doc._.rel_data.append(RelationData(sub=sub_offset, label=triple[2], objs=[(triple[3], triple[4])]))
        return doc
    
    def init_model(self, model_or_path):
        if isinstance(model_or_path, GPLinkerForRelationExtraction):
            self.model = model_or_path
            self.model.to(self.device)
            self.model.freeze()
            
        else:
            self.model= GPLinkerForRelationExtraction.load_from_checkpoint(model_or_path)
            self.model.to(self.device)
            self.model.freeze()
    
    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        # path : save_path/information_extractor
        if not os.path.exists(path):
            os.mkdir(path=path)
        model = 're.pkl'
        model_path = os.path.join(path, model)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
    def from_disk(self, path:str, exclude):
        model_path = os.path.join(path, 're.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        try:
            self.model.to(self.device)
        except:
            log.info(f' to device {self.device} failed')
        self.model.freeze()
        
class GlobalRelationExtractor():
    '''spacy组件 三元组抽取
    参数:
    - name: 组件名称
    - device: 设备
    '''
    def __init__(self, 
                 nlp, 
                 name:str, 
                 device:str, 
                 threshold:float,
                 prompts:List[str]):
        self.nlp = nlp
        self.pipe_name = name
        self.threshold = threshold
        self.device = device
        self.prompts = prompts
        
    def __call__(self, doc: Doc) -> Doc:
        texts = [doc.text for _ in range(len(self.prompts))]
        triples = self.model.predict(prompts=self.prompts, texts=texts, device=self.device, threshold=self.threshold)
        for triple in triples:
            sub_offset = (triple[0], triple[1])
            doc._.rel_data.append(RelationData(sub=sub_offset, label=triple[2], objs=[(triple[3], triple[4])]))
        return doc
    
    def init_model(self, model_or_path):
        if isinstance(model_or_path, GPLinkerForPromptRelationExtraction):
            self.model = model_or_path
            self.model.to(self.device)
            self.model.freeze()
            
        else:
            self.model= GPLinkerForPromptRelationExtraction.load_from_checkpoint(model_or_path)
            self.model.to(self.device)
            self.model.freeze()
    
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
        try:
            self.model.to(self.device)
        except:
            log.info(f' to device {self.device} failed')
        self.model.freeze()
    

default_config = {'threshold':0.0,
                  'prompts':[],
                  'device':'cpu',
                  'model':'global_relation'}

@Chinese.factory('relation_extractor',assigns=['doc._.relations'],default_config=default_config)
def make_relation_extractor(nlp, 
                          name:str, 
                          device:str, 
                          threshold:float,
                          model:str,
                          prompts:List[str]):
    """三元组抽取组件"""
    if model == 'gplinker':
        return GPLinkerExtractor(nlp=nlp, 
                                 name=name, 
                                 device=device, 
                                 threshold=threshold)
    elif model == 'global_relation':
        return GlobalRelationExtractor(nlp=nlp,
                                       name=name,
                                       device=device,
                                       threshold=threshold,
                                       prompts=prompts)



