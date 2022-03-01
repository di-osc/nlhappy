from spacy.lang.zh import Chinese
from ..models import BertCRF, BertTokenClassification
import shutil
from thinc.api import Config
import os
import torch

models = {
    'bert_crf': BertCRF, 
    'bert_token_classification': BertTokenClassification
    }

class SentTokencat:
    '''句子级别token分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, nlp, name:str, ckpt: str, model: str, device: str, ):
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.device = torch.device(device)
        self.model_name = model
        self.model_class = models[model]
        self.model = models[model].load_from_checkpoint(ckpt)
        self.model.to(self.device)
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

@Chinese.factory('senttoken_classifier',assigns=['doc.ents'],default_config={'model':'bert_crf', 'device':'cpu'})
def make_sent_tokencat(nlp, name, ckpt, model, device):
    return SentTokencat(nlp, name, ckpt, model, device)




class Tokencat:
    '''token分类spacy pipeline
    - ckpt: 模型保存路径
    '''
    def __init__(self, nlp, name:str, ckpt: str, model: str, device: str, ):
        self.nlp = nlp
        self.pipe_name = name
        self.ckpt = ckpt
        self.device = torch.device(device)
        self.model_name = model
        self.model_class = models[model]
        self.model = models[model].load_from_checkpoint(ckpt)
        self.model.to(self.device)
        self.model.freeze()
        
    def __call__(self, doc):
        all_spans = []
        spans = self.model.predict(doc.text, self.device)
        for span in spans:
            s = doc.char_span(span[0], span[1]+1, span[2])
            all_spans.append(s)
        doc.set_ents(all_spans)
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

@Chinese.factory('token_classifier',assigns=['doc.ents'],default_config={'model':'bert_crf', 'device':'cpu'})
def make_tokencat(nlp, name, ckpt, model, device):
    return Tokencat(nlp, name, ckpt, model, device)