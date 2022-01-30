import pytorch_lightning as pl
from typing import Optional,  Tuple, List
from transformers import BertTokenizer
from ..storers import OSSStorer
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
            pretrained_dir: str,
            data_dir: str 
            ):  
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)

        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.pretrained_model)
        self.label2id = None
        self.is_multi_label = is_multi_label
        self.max_length = max_length
        self.file_name = dataset+ '.zip'
        self.storage = OSSStorer()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        
        


    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        # 下载数据集
        if not os.path.exists(self.hparams.data_dir + self.hparams.dataset):
            self.storage.download(
                filename = self.file_name, 
                localfile = self.hparams.data_dir + self.file_name
                )
            with zipfile.ZipFile(file=self.hparams.data_dir + self.file_name, mode='r') as zf:
                zf.extractall(path=self.hparams.data_dir)
            os.remove(path=self.hparams.data_dir + self.file_name)
            
        #下载预训练模型
        if not os.path.exists(self.hparams.pretrained_dir + self.hparams.pretrained_model):
            self.storage.download(
                filename=self.pretrained_file,
                localfile=self.hparams.pretrained_dir + self.pretrained_file
            )
            with zipfile.ZipFile(file=self.hparams.pretrained_dir + self.pretrained_file, mode='r') as zf:
                zf.extractall(path=self.hparams.pretrained_dir)
            os.remove(path=self.hparams.pretrained_dir + self.pretrained_file)

    def set_transform(self, example) -> None:
        text = example['text'][0]
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens, 
            is_pretokenized=True,
            padding='max_length',  
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True)
        inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
        if self.is_multi_label:
            labels = example['labels'][0]
            label_ids = torch.tensor([self.label2id[label] for label in labels])
            return {'inputs': [inputs], 'label_ids': [label_ids]}
        else:
            label = example['label'][0]
            label_id = torch.tensor(self.label2id[label])
            return {'inputs': [inputs], 'label_ids': [label_id]} 
        

        
    def setup(self, stage: str) -> None:
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        if self.is_multi_label:
            set_labels = sorted(set([label for label in data['train']['labels']]))
        else:
            set_labels = sorted(set(data['train']['label']))
        self.label2id = {label: i for i, label in enumerate(set_labels)}
        data.set_transform(transform=self.set_transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']
        
            

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