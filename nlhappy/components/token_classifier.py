from spacy.lang.zh import Chinese
from ..models import BertCRF, BertTokenClassification
import shutil
from thinc.api import Config
import os
import torch
from spacy.tokens import Span

models = {
    'bert_crf': BertCRF, 
    'bert_token_classification': BertTokenClassification
    }

class Tokenclassifier:
    '''token分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, nlp, name:str, ckpt: str, model: str, device: str, sentence_level: bool):
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.sentence_level = sentence_level
        self.model_name = model
        self.device = torch.device(device)
        try:
            self.model = models[self.model_name].load_from_checkpoint(self.ckpt)
            self.model.to(self.device)
            self.model.freeze()
        except Exception:
            pass


        
    def __call__(self, doc):
        if self.sentence_level:
            all_spans = []
            for sent in doc.sents:
                text = ''.join([t.text for t in sent])
                spans = self.model.predict(text, self.device)
                for span in spans:
                    s = Span(doc, span[0]+sent.start, span[1]+1+sent.start, span[2])
                    all_spans.append(s)
            doc.set_ents(all_spans)
        else:
            all_spans = []
            text = ''.join([t.text for t in doc])
            spans = self.model.predict(text, self.device)
            for span in spans:
                s = Span(doc, span[0], span[1]+1, span[2])
                all_spans.append(s)
            doc.set_ents(all_spans)
        return doc
    
    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        if not os.path.exists(path):
            os.mkdir(path)
        shutil.copy(self.ckpt, path)
        # 重写NLP配置文件config.cfg 改变pipeline的ckpt路径
        # nlp_path = str(path).split(self.pipe_name)[0]
        # config_path = os.path.join(nlp_path, 'config.cfg')
        # config = Config().from_disk(config_path)
        # # config['components'][self.pipe_name]['ckpt_name'] = ckpt_name
        # config.to_disk(config_path)

    def from_disk(self, path:str, exclude):
        nlp_path = str(path).split(self.pipe_name)[0]
        config_path = os.path.join(nlp_path, 'config.cfg')
        config = Config().from_disk(config_path)
        ckpt_name = config['components'][self.pipe_name]['ckpt'].split('/')[-1]
        ckpt_path = os.path.join(path, ckpt_name)
        self.model = models[self.model_name].load_from_checkpoint(ckpt_path)
        self.model.freeze()

@Chinese.factory('token_classifier',assigns=['doc.ents'],default_config={'model':'bert_crf', 'device':'cpu'})
def make_tokencat(nlp, name, ckpt, model, device, sentence_level):
    return Tokenclassifier(nlp, name, ckpt, model, device, sentence_level)