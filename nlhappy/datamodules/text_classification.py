import pytorch_lightning as pl
from typing import Tuple, List, Dict
from transformers import AutoTokenizer, AutoConfig
from ..utils.make_datamodule import prepare_data_from_oss
from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch
import os
from ..utils import utils

log = utils.get_logger(__name__)

example = '''
        单标签:
        {'label': '新闻', 'text': '怎么给这个图片添加超级链接呢？'}
        多标签:
        {'labels': ['新闻', '财经'], 'text': '怎么给这个图片添加超级链接呢？'}
        '''

class TextClassificationDataModule(pl.LightningDataModule):
    '''
    文本分类
    '''
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int ,
                batch_size: int ,
                num_workers: int =0,
                pin_memory: bool =False,
                plm_dir: str ='./plms/',
                dataset_dir: str ='./datasets/'): 
        """单文本分类数据模块

        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            max_length (int): 文本最大长度
            batch_size (int): 批次大小
            num_workers (int, optional): _description_. Defaults to 0.
            pin_memory (bool, optional): _description_. Defaults to False.
            plm_dir (str, optional): _description_. Defaults to './plms/'.
            dataset_dir (str, optional): _description_. Defaults to './datasets/'.
        """
        
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)
        
        
        
        


    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        prepare_data_from_oss(dataset=self.hparams.dataset,
                              plm=self.hparams.plm,
                              dataset_dir=self.hparams.dataset_dir,
                              plm_dir=self.hparams.plm_dir)
        

    def transform(self, examples) -> Dict:
        batch_text = examples['text']
        batch_label = examples['label']
        batch = {'inputs':[], 'label_ids':[]}
        for i, text in enumerate(batch_text):
            inputs = self.tokenizer(
                text, 
                padding='max_length',  
                max_length=self.hparams.max_length+2,
                truncation=True)
            inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
            batch['inputs'].append(inputs)
            batch['label_ids'].append(self.hparams['label2id'][batch_label[i]])
        return batch
        

        
    def setup(self, stage: str) -> None:
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        self.dataset = load_from_disk(dataset_path)
        set_labels = set(self.dataset['train']['label'])
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams['label2id'] = label2id
        self.hparams['id2label'] = id2label
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        # self.hparams['vocab'] = dict(self.tokenizer.vocab)
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        trf_config = AutoConfig.from_pretrained(plm_path)
        self.hparams['trf_config'] = trf_config
        self.dataset.set_transform(transform=self.transform)
            

    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(
            dataset= self.dataset['train'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.dataset['validation'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.dataset['test'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)
        