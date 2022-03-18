import shutil
from ..models import BertGlobalPointer
from spacy.lang.zh import Chinese
from ..models import BertGlobalPointer
import torch
from spacy.tokens import Doc
from thinc.api import Config
import os

models = {'bert_global_pointer': BertGlobalPointer}

class SpanClassification:
    '''句子级别span分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, nlp, name:str, model:str, ckpt:str, device:str, sentence_level:bool):
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.sentence_level = sentence_level
        self.device = torch.device(device)
        self.model_name = model
        self.model_class = models[model]
        self.model = models[model].load_from_checkpoint(ckpt)
        self.model.to(self.device)
        self.model.freeze()
        
        
    def __call__(self, doc: Doc) -> Doc:
        all_spans = []
        if self.sentence_level:
            for sent in doc.sents:
                spans = self.model.predict(sent.text, device=self.device)
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

@Chinese.factory('span_classifier',assigns=['doc.spans'],default_config={'model':'bert_global_pointer', 'device':'cpu', 'sentence_level':False})
def make_spancat(nlp, name:str, model:str, ckpt:str, device:str, sentence_level:bool):
    """句子级别的文本片段分类"""
    return SpanClassification(nlp=nlp, name=name, model=model, ckpt=ckpt, device=device, sentence_level=sentence_level)

