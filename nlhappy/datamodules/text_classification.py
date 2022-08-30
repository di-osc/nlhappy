from functools import lru_cache
from typing import Tuple, List, Dict
from ..utils.make_datamodule import PLMBaseDataModule
from ..utils import utils
import torch

log = utils.get_logger(__name__)


class TextClassificationDataModule(PLMBaseDataModule):
    '''
    文本分类
    '''
    def __init__(self,
                dataset: str,
                plm: str,
                batch_size: int ,
                transform: str = 'simple',
                auto_length: str = 'max',
                num_workers: int =0,
                pin_memory: bool =False,
                plm_dir: str ='./plms/',
                dataset_dir: str ='./datasets/'): 
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
        
        super().__init__()        
        self.transforms = {'simple': self.simple_transform}
        assert self.hparams.transform in self.transforms.keys(), f'availabel models for text classification dm: {self.transforms.keys()}'
        
        
    def setup(self, stage: str) -> None:
        self.hparams.max_length = self.get_max_length()
        self.hparams.label2id = self.label2id
        self.hparams.id2label = {i:l for l, i in self.label2id.items()}
        self.dataset.set_transform(transform=self.simple_transform)

    
    def simple_transform(self, examples) -> Dict:
        batch_text = examples['text']
        batch_inputs = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[]}
        batch_label_ids = []
        max_length = self.hparams.max_length
        for i, text in enumerate(batch_text):
            inputs = self.tokenizer(text, 
                                    padding='max_length',  
                                    max_length=max_length,
                                    truncation=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_label_ids.append(self.hparams['label2id'][examples['label'][i]])
        batch_inputs['label_ids'] = batch_label_ids
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        
        return batch


    @property
    @lru_cache()
    def label2id(self):
        labels = set(self.train_df.label.values)
        label2id = {l : i for i,l in enumerate(labels)}
        return label2id

        
    @staticmethod
    def show_one_sample():
        return {'label': '新闻', 'text': '怎么给这个图片添加超级链接呢？'}
        