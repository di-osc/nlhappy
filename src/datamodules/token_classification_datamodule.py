import pytorch_lightning as pl
import os
from ..storers import OSSStorer
import zipfile
from datasets import load_from_disk
from transformers import BertTokenizer
import srsly
from torch.utils.data import DataLoader
import torch
from ..utils.preprocessing import fine_grade_tokenize


class TokenClassificationDataModule(pl.LightningDataModule):
    '''序列标注数据模块
    '''
    def __init__(self,
                dataset: str,
                pretrained_model: str,
                max_length: int,
                batch_size: int,
                pin_memory: bool,
                num_workers: int,
                pretrained_dir: str,
                storer: str = 'oss',
                data_dir: str = 'data/',
                label_pad_id: int = -100):
        super().__init__()
        self.pretrained_file = pretrained_model + '.zip'
        self.data_file = dataset + '.zip'
        self.storage = OSSStorer()
        self.tokenzier = None
        self.label2id = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.label_pad_id = label_pad_id
        self.max_length = max_length

        self.save_hyperparameters()
        
    def prepare_data(self):
        '''下载数据集和预训练模型'''
        if not os.path.exists(self.hparams.data_dir + self.hparams.dataset):
            self.storage.download(filename = self.data_file, 
                                localfile = self.hparams.data_dir + self.data_file)
            with zipfile.ZipFile(file=self.hparams.data_dir + self.data_file, mode='r') as zf:
                zf.extractall(path=self.hparams.data_dir)
            os.remove(path=self.hparams.data_dir + self.data_file)

        if not os.path.exists(self.hparams.pretrained_dir + self.hparams.pretrained_model):
            self.storage.download(filename = self.pretrained_file, 
                                localfile = self.hparams.pretrained_dir + self.pretrained_file)
            with zipfile.ZipFile(file=self.hparams.pretrained_dir + self.pretrained_file, mode='r') as zf:
                zf.extractall(path=self.hparams.pretrained_dir)
            os.remove(path=self.hparams.pretrained_dir + self.pretrained_file)

    def set_transform(self, example):
        tokens = example['tokens'][0]
        text = ''.join(tokens)
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens, 
            is_pretokenized=True, 
            add_special_tokens=True,
            padding='max_length',  
            max_length=self.max_length,
            truncation=True)
        inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
        labels = example['labels'][0]
        labels = [self.label2id[label] for label in labels] 
        labels = [self.label2id['O']] + labels + [self.label2id['O']] + (self.max_length - len(labels)-2) * [self.label_pad_id]
        labels = torch.tensor(labels)
        assert len(labels) == self.max_length
        return {'inputs':[inputs], 'label_ids':[labels]}



    def setup(self, stage):
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([label for labels in data['train']['labels'] for label in labels]))
        self.label2id = {label: i for i, label in enumerate(set_labels)}
        data.set_transform(transform=self.set_transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir +self.hparams.pretrained_model)
        


    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=True,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers)


    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers)        



    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(dataset=self.test_dataset,
                              batch_size=self.hparams.batch_size,
                              shuffle=False,
                              pin_memory=self.hparams.pin_memory,
                              num_workers=self.hparams.num_workers)




