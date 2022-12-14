from functools import lru_cache
from typing import Tuple, List, Dict
from ..utils.make_datamodule import PLMBaseDataModule
from ..utils import utils
import torch

log = utils.get_logger(__name__)


class TextClassificationDataModule(PLMBaseDataModule):
    """单文本分类数据模块
    
    Args:
        dataset (str): 数据集名称
        plm (str): 预训练模型名称
        batch_size (int): 批次大小
        transform (str): 数据转换形式,默认为simple
        num_workers (int, optional): 进程数. Defaults to 0.
        pin_memory (bool, optional): . Defaults to False.
        plm_dir (str, optional): 自定义预训练模型文件夹. Defaults to './plms/'.
        dataset_dir (str, optional): 自定义数据集文件夹. Defaults to './datasets/'.
        
    """
    def __init__(self,
                dataset: str,
                batch_size: int ,
                plm: str = 'hfl/chinese-roberta-wwm-ext',
                **kwargs):
        super().__init__()       
        
    def setup(self, stage: str) -> None:
        self.hparams.id2label = self.id2label

    
    def bert_transform(self, examples) -> Dict:
        batch_text = examples['text']
        batch_label_ids = []
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_inputs = self.tokenizer(batch_text,
                                      padding=True,
                                      max_length=max_length,
                                      truncation=True,
                                      return_tensors='pt')
        for i, text in enumerate(batch_text):
            label_id = self.label2id[examples['label'][i]]
            batch_label_ids.append(label_id)
        batch_inputs['label_ids'] = torch.LongTensor(batch_label_ids)        
        return batch_inputs
    
    
    @property
    @lru_cache()
    def labels(self):
        return sorted(set(self.train_df.label.values))


    @property
    def label2id(self):
        label2id = {l : i for i,l in enumerate(self.labels)}
        return label2id


    @property
    def id2label(self):
        return {i:l for l,i in enumerate(self.label2id)}

        
    @classmethod
    def get_one_example(cls):
        return {'label': '新闻', 'text': '怎么给这个图片添加超级链接呢？'}