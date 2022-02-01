import pytorch_lightning as pl
from typing import Optional,  Tuple, List
from transformers import AutoTokenizer
from ..storers import OSSStorer
from ..utils import utils
from torch.utils.data import DataLoader
import torch
import os
import zipfile
from datasets import load_from_disk

class SentencePairDataModule(pl.LightningDataModule):
    '''
    句子对数据模块,用来构建pytorch_lightning的数据模块
    - data_name: 数据集名称, 文件为datasets.DataSet  feature 必须包含 sentence1, sentence2, label
    - train_val_test_split: 训练集、验证集、测试集的比例
    - max_length: 句子的最大长度
    - return_sentence_pair: 是否返回句子对
    - batch_size: 批大小
    - num_workers: 加载数据时的线程数
    - pin_memory: 是否将数据加载到GPU
    - tokenizer_name: 分词器的名称
    '''
    def __init__(
            self,
            data_name: str,
            sentence_max_length: int ,
            tokenizer_name: str ='bert-base-chinese',
            return_sentence_pair: bool = False,
            batch_size: int = 64,
            num_workers: int = 2,
            pin_memory: bool =False,
            storer: str = 'oss',
            data_dir = 'data/'
    ):  
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
        self.file_name = data_name + '.zip'
        if self.hparams.storer == 'oss':
         self.storer = OSSStorer()
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        
        

    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        if not os.path.exists(self.hparams.data_dir + self.hparams.data_name):
            self.storer.download(filename = self.file_name, 
                                localfile = self.hparams.data_dir + self.file_name)
            with zipfile.ZipFile(self.hparams.data_dir + self.file_name) as zf:
                zf.extractall(self.hparams.data_dir)
        if os.path.exists(self.hparams.data_dir + self.file_name):
            os.remove(path=self.hparams.data_dir + self.file_name)
        
    
    def setup(self, stage: str):
        dataset = load_from_disk(self.hparams.data_dir + self.hparams.data_name)
        dataset.set_transform(transform=self.set_transform)
        self.test_dataset = dataset['test']
        split_ = dataset['train'].train_test_split(train_size=0.8)
        self.train_dataset = split_['train']
        self.validation_dataset = split_['test']
        
    def set_transform(self, example):
        if self.hparams.return_sentence_pair:
            inputs_1= self.tokenizer(example['sentence1'][0], 
                                     padding='max_length', 
                                     max_length=self.hparams.sentence_max_length, 
                                     truncation=True)
            inputs_1= dict(zip(inputs_1, map(torch.tensor, inputs_1.values())))
            inputs_2 = self.tokenizer(example['sentence2'][0], 
                                      padding='max_length', 
                                      max_length=self.hparams.sentence_max_length, 
                                      truncation=True)
            inputs_2 = dict(zip(inputs_2, map(torch.tensor, inputs_2.values())))
            label = torch.tensor(example['label'][0])
            return {'inputs_1': [inputs_1], 'inputs_2': [inputs_2], 'label': [label]}
        else:
            inputs = self.tokenizer(example['sentence1'][0], 
                                    example['sentence2'][0], 
                                    padding='max_length', 
                                    max_length=self.hparams.sentence_max_length*2, 
                                    truncation=True)
            inputs = dict(zip(inputs, map(torch.tensor, inputs.values())))
            label = torch.tensor(example['label'][0])
            return {'inputs': [inputs], 'label': [label]}

    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(dataset= self.train_dataset, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(dataset=self.validation_dataset, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(dataset=self.test_dataset, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)


        