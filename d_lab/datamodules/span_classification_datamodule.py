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
        data_dir: str = './data/',
        pretrained_dir: str = './pretrained_models/'
        ) :
        super().__init__()
        self.save_hyperparameters()


    def prepare_data(self) -> None:
        '下载数据集和预训练模型'
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.data_dir)
        oss.download_model(self.hparams.pretrained_model, self.hparams.pretrained_dir)

    def set_transform(self, example):
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
        spans = example['spans'][0]
        span_ids = torch.zeros(len(self.hparams.label2id), self.hparams.max_length, self.hparams.max_length)
        for span in spans :
            # +1 是因为添加了 [CLS]
            span_ids[self.hparams.label2id[span[2]],  int(span[0])+1, int(span[1])+1] = 1

        return {'inputs':[inputs], 'span_ids':[span_ids]}

    def setup(self, stage: str) -> None:
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([span[2] for spans in data['train']['spans'] for span in spans]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams.label2id = label2id
        self.hparams.id2label = id2label
        data.set_transform(transform=self.set_transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']
        else: self.test_dataset = None
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.pretrained_model)
        self.hparams.token2id = dict(self.tokenizer.vocab)
    # def collate_fn(self, batch):
    #     new = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[], 'span_ids':[]}
    #     for e in batch:
    #         for k, v in e['inputs'].items():
    #             new[k].append(v)
    #         new['span_ids'].append(e['span_ids'])
    #     new_ = dict(zip(new.keys(), map(torch.stack, new.values())))
    #     res = (new_['input_ids'], new_['token_type_ids'], new_['attention_mask']), new_['span_ids'].to_sparse()
    #     return res


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
                              num_workers=self.hparams.num_workers
                              )