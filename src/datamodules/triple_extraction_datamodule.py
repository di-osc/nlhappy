import pytorch_lightning as pl
from ..storers import OSSStorer
import zipfile
import os
from ..utils.preprocessing import fine_grade_tokenize
import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from torch.utils.data import DataLoader

class TripleExtractionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        pretrained_model: str,
        max_length: int ,
        batch_size: int ,
        num_workers: int ,
        pin_memory: bool ,
        pretrained_dir: str,
        data_dir: str 
        ):  
        super().__init__()

        self.save_hyperparameters()

        self.max_length = max_length
        self.file_name = dataset+ '.zip'
        self.pretrained_file = pretrained_model + '.zip'
        self.storage = OSSStorer()


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

        triples = example['triples'][0]
        span_ids = torch.zeros(2, self.max_length, self.max_length)
        head_ids = torch.zeros(len(self.label2id), self.max_length, self.max_length)
        tail_ids = torch.zeros(len(self.label2id), self.max_length, self.max_length)
        for triple in triples:
            #加1是因为有cls
            span_ids[0][triple['subject_index'][0]+1][triple['subject_index'][1]+1] = 1
            span_ids[1][triple['object_index'][0]+1][triple['object_index'][1]+1] = 1
            head_ids[self.label2id[triple['predicate']]][triple['subject_index'][0]+1][triple['object_index'][0]+1] = 1
            tail_ids[self.label2id[triple['predicate']]][triple['subject_index'][1]+1][triple['object_index'][1]+1] = 1

        return {'inputs': [inputs], 'span_ids': [span_ids], 'head_ids': [head_ids], 'tail_ids': [tail_ids]} 

    def setup(self, stage: str) -> None:
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([triple['predicate'] for triples in data['train']['triples'] for triple in triples]))
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