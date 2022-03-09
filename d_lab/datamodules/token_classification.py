import pytorch_lightning as pl
import os
from ..utils.storer import OSSStorer
import zipfile
from datasets import load_from_disk
from transformers import BertTokenizer, BertConfig
import srsly
from torch.utils.data import DataLoader
import torch
from ..utils.preprocessing import fine_grade_tokenize


class TokenClassificationDataModule(pl.LightningDataModule):
    '''序列标注数据模块
    数据集字段:
    - tokens: list[str], 例如: ['我', '是', '一', '中', '国', '人', '。']
    - labels: list[str], 例如: ['O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']
    '''
    def __init__(
        self,
        dataset: str,
        plm: str,
        max_length: int,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        pretrained_dir: str ,
        data_dir: str ,
        label_pad_id: int = -100
        ):
        super().__init__()

        self.save_hyperparameters()

        
        
    def prepare_data(self):
        '''下载数据集和预训练模型'''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.data_dir)
        oss.download_plm(self.hparams.pretrained_model, self.hparams.pretrained_dir)

    def transform(self, example):
        tokens = example['tokens'][0]
        text = ''.join(tokens)
        new_tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            new_tokens, 
            is_pretokenized=True, 
            add_special_tokens=True,
            padding='max_length',  
            max_length=self.hparams.max_length,
            truncation=True)
        inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
        labels = example['labels'][0]
        labels = [self.hparams.label2id[label] for label in labels] 
        labels = [self.hparams.label2id['O']] + labels + [self.hparams.label2id['O']] + (self.hparams.max_length - len(labels)-2) * [self.hparams.label_pad_id]
        labels = torch.tensor(labels)
        return {'inputs':[inputs], 'label_ids':[labels]}



    def setup(self, stage):
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([label for labels in data['train']['labels'] for label in labels]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams.label2id = label2id
        self.hparams.id2label = id2label
        data.set_transform(transform=self.transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir +self.hparams.pretrained_model)
        self.hparams.token2id = dict(self.tokenizer.vocab)
        bert_config = BertConfig.from_pretrained(self.hparams.pretrained_dir + self.hparams.pretrained_model)
        self.hparams.bert_config = bert_config
        


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




