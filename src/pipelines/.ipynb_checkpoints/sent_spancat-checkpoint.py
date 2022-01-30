from ..models import BERTGlobalPointer
from spacy.lang.zh import Chinese


class SentSpancat:
    '''句子级别span分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, ckpt: str):
        self.model = BERTGlobalPointer.load_from_checkpoint(ckpt)
        self.model.freeze()
        
    def __call__(self, doc):
        all_spans = []
        for sent in doc.sents:
            spans = self.model.predict(sent.text)
            for span in spans:
                s = sent.char_span(span[0], span[1]+1, span[2])
                all_spans.append(s)
        doc.spans['all'] = all_spans
        return doc

@Chinese.factory('sent_spancat',assigns=['doc.sents'],default_config={'ckpt':str})
def make_sent_spancat(nlp, name, ckpt):
    return SentSpancat(ckpt)