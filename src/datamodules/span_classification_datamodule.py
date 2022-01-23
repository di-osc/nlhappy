import pytorch_lightning as pl
from ..storers import OSSStorer
import os
import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from ..utils.preprocessing import fine_grade_tokenize
import zipfile


class SpanClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        pretrained_model: str,
        max_length: int,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        data_dir: str,
        pretrained_dir: str,
        storer: str = 'oss'
    ) :
        super().__init__()
        self.save_hyperparameters()

        self.file_name = dataset + '.zip'
        self.pretrained_file = pretrained_model + '.zip'
        if storer == 'oss':
            self.storage = OSSStorer()
        self.max_length = max_length
        self.tokenizer = None
        self.label2id = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None


    def prepare_data(self) -> None:
        '下载数据集和预训练模型'
        if not os.path.exists(self.hparams.data_dir + self.hparams.dataset):
            self.storage.download(filename = self.file_name, 
                                 localfile = self.hparams.data_dir + self.file_name)
            with zipfile.ZipFile(file=self.hparams.data_dir + self.file_name, mode='r') as zf:
                zf.extractall(path=self.hparams.data_dir)
            os.remove(path=self.hparams.data_dir + self.file_name)
        if not os.path.exists(self.hparams.pretrained_dir + self.hparams.pretrained_model):
            self.storage.download(
                file_name=self.pretrained_file,
                localfile=self.hparams.pretrained_dir + self.pretrained_file
            )
            with zipfile.ZipFile(file=self.hparams.pretrained_dir + self.pretrained_file, mode='r') as zf:
                zf.extractall(path=self.hparams.pretrained_dir)
            os.remove(path=self.hparams.pretrained_dir + self.pretrained_file)

    def set_transform(self, example):
        text = example['text'][0]
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens, 
            is_pretokenized=True,
            padding='max_length',  
            max_length=self.max_length,
            truncation=True)
        inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
        spans = example['spans'][0]
        span_ids = torch.zeros(len(self.label2id), self.max_length, self.max_length)
        for span in spans :
            span_ids[self.label2id[span[2]],  int(span[0])+1, int(span[1])+1] = 1
        return {'inputs':[inputs], 'span_ids':[span_ids]}

    def setup(self, stage: str) -> None:
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([span[2] for spans in data['train']['spans'] for span in spans]))
        self.label2id = {label: i for i, label in enumerate(set_labels)}
        data.set_transform(transform=self.set_transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.pretrained_model)


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