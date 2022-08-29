import torch
import pytorch_lightning as pl
from typing import List
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
import os
from ..utils.make_datamodule import prepare_data_from_remote



class TextPairRegressionDataModule(pl.LightningDataModule):
    '''文本对相似度数据模块
    dataset_exmaple:
        {'text_a': '左膝退变伴游离体','text_b': '单侧膝关节骨性关节病','similarity': 0}
    '''
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int,
                batch_size: int,
                num_workers: int = 0,
                pin_memory: bool =False,
                dataset_dir = './datasets/',
                plm_dir = './plms/'):
        """
        Args:
            dataset (str): the name of the dataset.
            plm (str): the name of the plm.
            max_length (int): the max length of the text
            batch_size (int): the batch size in training and validation.
            num_workers (int, optional): num of workers for data loading. 0 means that the data will be loaded in the main process. Defaults to 0.
            pin_memory (bool, optional): whether to use pin memory. Defaults to False.
            dataset_dir (str, optional): the directory of the dataset. Defaults to './datasets'.
            plm_dir (str, optional): the directory of the plm. Defaults to './plms'.
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

        
    def setup(self, stage: str):
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        self.dataset = load_from_disk(dataset_path)
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        self.hparams['trf_config'] = AutoConfig.from_pretrained(plm_path)
        self.dataset.set_transform(transform=self.transform)

    def transform(self, examples):
        batch_text_a = examples['text_a']
        batch_text_b = examples['text_b']
        similarities = examples['similarity']
        batch = {'inputs_a': [], 'inputs_b': [], 'similarities':[]}
        for i  in range(len(batch_text_a)):
            inputs_a= self.tokenizer(batch_text_a[i], 
                                    padding='max_length', 
                                    max_length=self.hparams.max_length, 
                                    truncation=True)
            inputs_a = dict(zip(inputs_a.keys(), map(torch.tensor, inputs_a.values())))
            batch['inputs_a'].append(inputs_a)
            inputs_b = self.tokenizer(batch_text_b[i],
                                    padding='max_length', 
                                    max_length=self.hparams.max_length, 
                                    truncation=True)
            inputs_b = dict(zip(inputs_b.keys(), map(torch.tensor, inputs_b.values())))
            batch['inputs_b'].append(inputs_b)
            batch['similarities'].append(torch.tensor(similarities[i], dtype=torch.float))
        
        return batch
        


    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(dataset= self.dataset['train'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(dataset=self.dataset['validation'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(dataset=self.dataset['test'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)
