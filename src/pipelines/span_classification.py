import shutil
from ..models import BertGlobalPointer
from spacy.lang.zh import Chinese
from ..models import BertGlobalPointer
import torch
from spacy.tokens import Doc

models = {'bert_global_pointer': BertGlobalPointer}

class SentSpancat:
    '''句子级别span分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, nlp, name, model_name: str, ckpt: str, device: str):
        self.nlp = nlp
        self.pipe_name = name
        self.model_class = models[model_name]
        self.ckpt = ckpt
        self.device = torch.device(device)
        self.model = models[model_name].load_from_checkpoint(ckpt)
        
        
    def __call__(self, doc: Doc) -> Doc:
        self.model.to(self.device)
        self.model.freeze()
        self.model.eval()
        all_spans = []
        for sent in doc.sents:
            spans = self.model.predict(sent.text)
            for span in spans:
                s = sent.char_span(span[0], span[1]+1, span[2])
                all_spans.append(s)
        doc.spans['all'] = all_spans
        return doc

    def to_disk(self, path:str, exclude):
        shutil.copy(self.ckpt, path)
        self.nlp.config['components'][self.pipe_name]['ckpt'] = path

    def from_disk(self, path:str, exclude):
        self.model = self.model_class.load_from_checkpoint(path)

@Chinese.factory('sent_span_classification',assigns=['doc.spans'],default_config={'model_name':'bert_global_pointer', 'device':'cpu'})
def make_sent_spancat(nlp, name:str, model_name:str, ckpt:str, device:str):
    """句子级别的文本片段分类"""
    return SentSpancat(nlp=nlp, name=name, model_name=model_name, ckpt=ckpt, device=device)

