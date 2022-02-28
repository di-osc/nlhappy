import pytorch_lightning as pl
from ..utils.storer import OSSStorer
from ..utils.preprocessing import fine_grade_tokenize
import torch
from datasets import load_from_disk
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from typing import Dict

class TripleExtractionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        pretrained_model: str,
        max_length: int ,
        batch_size: int ,
        num_workers: int ,
        pin_memory: bool =True,
        data_dir: str ='./data/',
        pretrained_dir: str  = './pretrained_models/'
        ):  
        super().__init__()

        self.save_hyperparameters()


    def prepare_data(self):
        '''下载数据集和预训练模型.'''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, localpath=self.hparams.data_dir)
        oss.download_model(self.hparams.pretrained_model, localpath=self.hparams.pretrained_dir)

    
    def setup(self, stage: str) -> None:
        """读取数据集, 对数据设置转换"""
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([triple['predicate'] for triples in data['train']['triples'] for triple in triples]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams['label2id'] = label2id
        self.hparams['id2label'] = id2label
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.pretrained_model)
        self.bert_config = BertConfig.from_pretrained(self.hparams.pretrained_dir + self.hparams.pretrained_model)
        self.hparams['bert_config'] = self.bert_config
        data.set_transform(transform=self.transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']


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
        triples = example['triples'][0]
        span_ids = torch.zeros(2, self.hparams.max_length, self.hparams.max_length)
        head_ids = torch.zeros(len(self.hparams.label2id), self.hparams.max_length, self.hparams.max_length)
        tail_ids = torch.zeros(len(self.hparams.label2id), self.hparams.max_length, self.hparams.max_length)
        for triple in triples:
            #加1是因为有cls
            span_ids[0][triple['subject_index'][0]+1][triple['subject_index'][1]+1] = 1
            span_ids[1][triple['object_index'][0]+1][triple['object_index'][1]+1] = 1
            head_ids[self.hparams.label2id[triple['predicate']]][triple['subject_index'][0]+1][triple['object_index'][0]+1] = 1
            tail_ids[self.hparams.label2id[triple['predicate']]][triple['subject_index'][1]+1][triple['object_index'][1]+1] = 1
        return {'inputs': [inputs], 'span_ids': [span_ids], 'head_ids': [head_ids], 'tail_ids': [tail_ids]}


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
        return DataLoader(dataset=self.test_dataset,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            pin_memory=self.hparams.pin_memory,
                            num_workers=self.hparams.num_workers)