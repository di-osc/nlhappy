import pytorch_lightning as pl
from typing import Optional,  Tuple, List, Dict
from transformers import BertTokenizer
from ..utils.storer import OSSStorer
import pandas as pd
from ..utils import utils
from torch.utils.data import DataLoader
import os
from ..utils.preprocessing import fine_grade_tokenize
import zipfile
from datasets import load_from_disk
import torch

class TextClassificationDataModule(pl.LightningDataModule):
    '''
    文本分类
    '''
    def __init__(
        self,
        dataset: str,
        pretrained_model: str,
        max_length: int ,
        is_multi_label: bool,
        batch_size: int ,
        num_workers: int ,
        pin_memory: bool ,
        pretrained_dir: str ,
        data_dir: str ,
        label_pad_id: int , 
        ):  
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)
        
        


    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, localpath=self.hparams.data_dir)
        oss.download_model(self.hparams.pretrained_model, localpath=self.hparams.pretrained_dir)
        

    def transform(self, example) -> Dict:
        text = example['text'][0]
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens, 
            is_pretokenized=True,
            padding='max_length',  
            max_length=self.hparams.max_length,
            add_special_tokens=True,
            truncation=True)
        inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
        if self.hparams.is_multi_label:
            labels = example['labels'][0]
            label_ids = torch.zeros(len(self.hparams.label2id))
            for label in labels:
                label_ids[self.hparams.label2id[label]] = 1
            return {'inputs': [inputs], 'label_ids': [label_ids]}
        else:
            label = example['label'][0]
            label_id = torch.tensor(self.hparams.label2id[label])
            return {'inputs': [inputs], 'label_ids': [label_id]} 
        

        
    def setup(self, stage: str) -> None:
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        if self.hparams.is_multi_label:
            set_labels = sorted(set([label for labels in data['train']['labels'] for label in labels]))
        else:
            set_labels = sorted(set(data['train']['label']))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams['label2id'] = label2id
        self.hparams['id2label'] = id2label
        data.set_transform(transform=self.transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.pretrained_model)
        
            

    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(
            dataset= self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.valid_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)