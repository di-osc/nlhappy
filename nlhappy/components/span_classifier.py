from ..models import GlobalPointer
from spacy.lang.zh import Chinese
import torch
from spacy.tokens import Doc
import os
from spacy.util import filter_spans
import pickle
from nlhappy.utils.utils import get_logger


log = get_logger()

class SpanClassification:
    '''句子级别span分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, nlp, name:str, device:str, sentence_level:bool, threshold:float, set_ents: bool):
        self.nlp = nlp
        self.pipe_name = name
        self.set_ents = set_ents
        self.sentence_level = sentence_level
        self.device = torch.device(device)
        self.threshold = threshold
        
        
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
        if not os.path.exists(path):
            os.mkdir(path=path)
        model = 'sc.pkl'
        model_path = os.path.join(path, model)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def from_disk(self, path:str, exclude):
        model_path = os.path.join(path, 'sc.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model.freeze()
        try:
            self.model.to(self.device)
        except:
            log.info(f' to device {self.device} failed')
            
    def init_model(self, model_or_path):
        if isinstance(model_or_path, GlobalPointer):
            self.model = model_or_path
            self.model.freeze()
            self.model.to(self.device)
            
        else:
            self.model= GlobalPointer.load_from_checkpoint(model_or_path)
            self.model.freeze()
            self.model.to(self.device)
            
default_config={'device':'cpu', 'sentence_level':False, 'threshold':0.5,'set_ents':True}

@Chinese.factory('span_classifier',assigns=['doc.spans'],default_config=default_config)
def make_spancat(nlp, name:str, device:str, sentence_level:bool, threshold:float, set_ents: bool):
    """句子级别的文本片段分类"""
    return SpanClassification(nlp=nlp, name=name, device=device, sentence_level=sentence_level, threshold=threshold, set_ents=set_ents)

