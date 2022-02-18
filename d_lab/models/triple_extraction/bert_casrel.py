import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from ...layers import SimpleDense, EfficientGlobalPointer


class BertCasRel(pl.LightningModule):
    """三元组抽取模型
    思路: 先找到主语, 然后去找每个关系下的宾语
    参考: 
    - https://github.com/xiangking/ark-nlp/blob/main/ark_nlp/model/re/casrel_bert/casrel_bert.py
    - 
    """
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        dropout: float,
        weight_decay: float,
        **data_params
    ):
        super(BertCasRel, self).__init__()
        self.save_hyperparameters()
        
        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.tokenizer = BertTokenizer.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])

        self.sub_head_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, 1)
        self.sub_tail_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, 1)
        self.obj_head_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(data_params['label2id']))
        self.obj_tail_classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(data_params['label2id']))

    


    


        

        
