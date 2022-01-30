from spacy.lang.zh import Chinese
from ..models import BertCRF

class SentTokencat:
    '''句子级别token分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, ckpt: str):
        self.model = BertCRF.load_from_checkpoint(ckpt)
        self.model.freeze()
        
    def __call__(self, doc):
        all_spans = []
        for sent in doc.sents:
            spans = self.model.predict(sent.text)
            for span in spans:
                s = sent.char_span(span[0], span[1]+1, span[2])
                all_spans.append(s)
        doc.set_ents(all_spans)
        return doc

@Chinese.factory('sent_tokencat',assigns=['doc.sents'],default_config={'ckpt':str})
def make_sent_tokencat(nlp, name, ckpt):
    return SentTokencat(ckpt)