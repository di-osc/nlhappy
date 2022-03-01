import shutil
from ..models import BertGlobalPointer
from spacy.lang.zh import Chinese
from ..models import BertGPLinker
import torch
from spacy.tokens import Doc
from thinc.api import Config
import os

models = {'bert_gplinker': BertGPLinker}
Doc.set_extension('triples', default=[])

class TripleExtractor(object):
    '''spacy组件 三元组抽取
    参数:
    - name: 组件名称
    - ckpt: pl模型保存路径
    - model: pl模型名称
    - device: 设备
    '''
    def __init__(self, nlp, name:str, model:str, ckpt:str, device:str):
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.device = torch.device(device)
        self.model_name = model
        self.model_class = models[model]
        self.model = models[model].load_from_checkpoint(ckpt)
        self.model.to(self.device)
        self.model.freeze()
        
        
    def __call__(self, doc):
        triples = self.model.predict(doc.text, self.device)
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

@Chinese.factory('triple_extractor',assigns=['doc._.triples'],default_config={'model':'bert_gplinker', 'device':'cpu'})
def make_triple_extractor(nlp, name:str, model:str, ckpt:str, device:str):
    """三元组抽取组件"""
    return TripleExtractor(nlp=nlp, name=name, model=model, ckpt=ckpt, device=device)