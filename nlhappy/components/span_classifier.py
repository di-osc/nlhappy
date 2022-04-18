import shutil
from ..models import BertGlobalPointer
from spacy.lang.zh import Chinese
from ..models import BertGlobalPointer
import torch
from spacy.tokens import Doc
from thinc.api import Config
import os
from spacy.util import filter_spans

models = {'bert_global_pointer': BertGlobalPointer}

class SpanClassification:
    '''句子级别span分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, nlp, name:str, model:str, ckpt:str, device:str, sentence_level:bool, threshold:float, set_ents: bool):
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.set_ents = set_ents
        self.sentence_level = sentence_level
        self.device = torch.device(device)
        self.threshold = threshold
        self.model_name = model
        try:
            self.model = models[self.model_name].load_from_checkpoint(self.ckpt)
            self.model.to(self.device)
            self.model.freeze()
        except Exception:
            pass
        
        
        
    def __call__(self, doc: Doc) -> Doc:
        all_spans = []
        if self.sentence_level:
            for sent in doc.sents:
                spans = self.model.predict(sent.text, device=self.device, threshold=self.threshold)
                for span in spans:
                    s = sent.char_span(span[0], span[1], span[2])
                    all_spans.append(s)
            doc.spans['all'] = all_spans
        else:
            spans = self.model.predict(doc.text, device=self.device)
            for span in spans:
                s = doc.char_span(span[0], span[1], span[2])
                all_spans.append(s)
            doc.spans['all'] = all_spans
        if self.set_ents:
            doc.set_ents(filter_spans(doc.spans['all']))
        return doc

    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        shutil.copy(self.ckpt, path)
        # 重写NLP配置文件config.cfg 改变pipeline的ckpt路径
        # nlp_path = str(path).split(self.pipe_name)[0]
        # config_path = os.path.join(nlp_path, 'config.cfg')
        # config = Config().from_disk(config_path)
        # config['components'][self.pipe_name]['ckpt'] = str(path)
        # config.to_disk(config_path)

    def from_disk(self, path:str, exclude):
        nlp_path = str(path).split(self.pipe_name)[0]
        config_path = os.path.join(nlp_path, 'config.cfg')
        config = Config().from_disk(config_path)
        ckpt_name = config['components'][self.pipe_name]['ckpt'].split('/')[-1]
        ckpt_path = os.path.join(path, ckpt_name)
        self.model = models[self.model_name].load_from_checkpoint(ckpt_path)
        self.model.freeze()

@Chinese.factory('span_classifier',assigns=['doc.spans'],default_config={'model':'bert_global_pointer', 'device':'cpu', 'sentence_level':False, 'threshold':0.5,'set_ents':False})
def make_spancat(nlp, name:str, model:str, ckpt:str, device:str, sentence_level:bool, threshold:float, set_ents: bool):
    """句子级别的文本片段分类"""
    return SpanClassification(nlp=nlp, name=name, model=model, ckpt=ckpt, device=device, sentence_level=sentence_level, threshold=threshold, set_ents=set_ents)

