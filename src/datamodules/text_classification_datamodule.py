import pytorch_lightning as pl
from typing import Optional,  Tuple, List
from transformers import AutoTokenizer
from ..storers import BaiduPanStorer, OSSStorer
import pandas as pd
from ..utils import utils
from torch.utils.data import DataLoader
import os

class TextClassificationDataModule(pl.LightningDataModule):
    '''
    文本分类
    '''
    def __init__(
            self,
            data_name: str,
            train_val_test_split: List[float] = [0.8, 0.1, 0.1],
            tokenizer_name: str = 'bert-base-chinese',
            text_max_length: int = 50,
            is_multi_label: bool = False,
            batch_size: int = 64,
            num_workers: int = 10,
            pin_memory: bool =False,
            data_dir: str = 'data/',
            storer: str = 'oss'
            ):  
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
        self.file_name = data_name + '.zip'
        if self.hparams.storer == 'oss':
            self.storer = OSSStorer()
        self.trainset = None
        self.valset = None
        self.testset = None
        
        


    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        if not os.path.exists(self.hparams.data_dir + self.file_name):
            self.storer.download(
                filename = self.file_name, 
                localfile = self.hparams.data_dir + self.file_name
                )
        
    
    def setup(self, stage: str):
        #切分数据集
        data_df = pd.read_csv(self.hparams.data_dir + self.file_name)
        train_df = data_df[data_df['text'].isin(data_df['text'].drop_duplicates().sample(frac=self.hparams.train_val_test_split[0]))]
        frac = self.hparams.train_val_test_split[1] / (self.hparams.train_val_test_split[1] + self.hparams.train_val_test_split[2])
        sub_df = data_df[~data_df['text'].isin(train_df)]
        val_df = sub_df.drop_duplicates(subset=['text']).sample(frac=frac)
        test_df = sub_df[~sub_df['text'].isin(val_df)]

        #保存数据集
        train_df.to_csv('train.csv', index=False)
        val_df.to_csv('val.csv', index=False)
        test_df.to_csv('test.csv', index=False)

        #构建dataset
        self.trainset = TextClassificationDataset(
            df=train_df, 
            tokenizer=self.tokenizer, 
            text_max_length=self.hparams.text_max_length,
            is_multi_label=self.hparams.is_multi_label)
        self.valset = TextClassificationDataset(
            df=val_df, 
            tokenizer=self.tokenizer, 
            text_max_length=self.hparams.text_max_length,
            is_multi_label=self.hparams.is_multi_label)
        self.testset = TextClassificationDataset(
            df=test_df, 
            tokenizer=self.tokenizer, 
            text_max_length=self.hparams.text_max_length,
            is_multi_label=self.hparams.is_multi_label)

    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(
            dataset= self.trainset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.valset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.testset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)