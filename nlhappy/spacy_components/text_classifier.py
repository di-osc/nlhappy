from spacy.lang.zh import Chinese
from ..models import BertTextClassification
import torch
import shutil
from spacy.tokens import Doc

models = {'bert_text_classification': BertTextClassification}

Doc.set_extension('labels', default=set())

class TextClassification:
    '''spacy 文本分类管道'''
    def __init__(self, nlp, name:str, model_name: str, ckpt: str, device: str):
        super().__init__()
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.device = torch.device(device)
        self.model_class = models[model_name]
        self.model = models[model_name].load_from_checkpoint(ckpt)

        
    def __call__(self, doc: Doc) -> Doc:
        with torch.no_grad():
            doc.cats = self.model.predict(doc.text, device=self.device)
        return doc

    def to_disk(self, path: str, exclude):
        shutil.copy(self.ckpt, path)
        self.nlp.config['components'][self.pipe_name]['ckpt'] = path

    def from_disk(self, path: str, exclude):
        self.model = self.model_class.load_from_checkpoint(path)


@Chinese.factory('text_classifier', default_config={'model_name':'bert_text_classification'})
def make_text_classification(nlp, name:str, model_name: str, ckpt: str, device: str):
    return TextClassification(nlp, name, model_name, ckpt, device)


    