from spacy.lang.zh import Chinese
from ..models import BertTextClassification
import torch
from spacy.tokens import Doc
import os
import pickle
import logging

log = logging.getLogger()


Doc.set_extension('labels', default=[])

class BertTextClassifier:
    '''spacy 文本分类管道'''
    def __init__(self, 
                 nlp, 
                 name:str, 
                 device: str,
                 threshold: float):
        super().__init__()
        self.nlp = nlp
        self.pipe_name = name
        self.device = torch.device(device)
        self.threshold = threshold
        
    def __call__(self, doc: Doc) -> Doc:
        with torch.no_grad():
            scores = self.model.predict(doc.text, device=self.device)
            if scores[0][1] >= self.threshold:
                doc._.label = scores[0][0]
            doc.cats = dict(scores)
        return doc
    
    def init_model(self, model_or_path):
        if isinstance(model_or_path, BertTextClassification):
            self.model = model_or_path
            self.model.to(self.device)
            self.model.freeze()
            
        else:
            self.model= BertTextClassification.load_from_checkpoint(model_or_path)
            self.model.to(self.device)
            self.model.freeze()

    def to_disk(self, path:str, exclude):
        if not os.path.exists(path):
            os.mkdir(path=path)
        model = 'tc.pkl'
        model_path = os.path.join(path, model)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
    def from_disk(self, path:str, exclude):
        model_path = os.path.join(path, 'tc.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        try:
            self.model.to(self.device)
        except:
            log.info(f' to device {self.device} failed')
        self.model.freeze()

default_config = {'model': 'bert_tc', 'device':'cpu', 'threshold':0.8}
@Chinese.factory('text_classifier', default_config=default_config)
def make_text_classification(nlp, name:str, device:str, model: str, threshold: float):
    if model == 'bert_tc':
        return BertTextClassifier(nlp, name=name, device=device, threshold=threshold)


    