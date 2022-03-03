import shutil
from d_lab.models import BertGlobalPointer, BertGPLinker
from spacy.lang.zh import Chinese
import torch
from spacy.tokens import Doc, Span
from thinc.api import Config
import os
from typing import Union, Optional, List, Tuple
from ..utils.type import SO, SPO

models = {'bert_gplinker': BertGPLinker}
Doc.set_extension('spoes', default=[])


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
        
        
    def __call__(self, doc):
        triples = self.model.predict(doc.text, self.device, threshold=self.threshold)
        for triple in triples:
            sub = SO(doc, triple[0], triple[1])
            predicate = triple[2]
            obj = SO(doc, triple[3], triple[4])
            spo = (sub, predicate, obj)
            doc._.spoes.append(SPO(spo))
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

@Chinese.factory('triple_extractor',assigns=['doc._.spoes'],default_config={'model':'bert_gplinker', 'device':'cpu', 'threshold':None})
def make_triple_extractor(nlp, name:str, model:str, ckpt:str, device:str, threshold):
    """三元组抽取组件"""
    return TripleExtractor(nlp=nlp, name=name, model=model, ckpt=ckpt, device=device, threshold=threshold)


if __name__ == "__main__":
    import spacy
    nlp = spacy.blank('zh')
    doc = nlp('歌曲《墨写你的美》是由歌手冷漠演唱的一首歌曲')
    doc.set_ents([Span(doc, 3, 8, "歌曲")])
    sub = SO(doc, 3, 8)
    print(sub)