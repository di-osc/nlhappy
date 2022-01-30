from spacy.lang.zh import Chinese
from ..models import BertTextClassification
import torch

class TextClassification:
    def __init__(self, ckpt: str, device: str='cpu'):
        super().__init__()
        self.model = BertTextClassification.load_from_checkpoint(ckpt)
        self.model.freeze()
        self.device = torch.device(device)
        self.model.to(self.device)

    def __call__(self, doc):
        doc.cats = self.model.predict(doc.text)
        return doc


@Chinese.factory('text_classification',default_config={'ckpt':str, 'device':str})
def make_text_classification(nlp, name, ckpt, device):
    return TextClassification(ckpt, device)