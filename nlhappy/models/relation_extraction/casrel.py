import pytorch_lightning as pl
from transformers import AutoModel
from ...utils.make_model import get_hf_tokenizer
from ...layers import SimpleDense, EfficientGlobalPointer


class CasRel(pl.LightningModule):
    """三元组抽取模型
    思路: 先找到主语, 然后去找每个关系下的宾语
    参考: 
    - https://github.com/xiangking/ark-nlp/blob/main/ark_nlp/model/re/casrel_bert/casrel_bert.py
    - 
    """
    def __init__(self,
                 hidden_size: int,
                 lr: float,
                 dropout: float,
                 weight_decay: float,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.bert = AutoModel.from_config(self.hparams.trf_config)
        self.tokenizer = get_hf_tokenizer(config=self.hparams.trf_config, vocab=self.hparams.vocab)

        self.sub_head_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, 1)
        self.sub_tail_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, 1)
        self.obj_head_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(kwargs['label2id']))
        self.obj_tail_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(kwargs['label2id']))

    


    


        

        
