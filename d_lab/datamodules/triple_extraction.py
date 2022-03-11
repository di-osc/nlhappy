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
        plm: str,
        max_length: int ,
        batch_size: int ,
        num_workers: int ,
        pin_memory: bool,
        data_dir: str ,
        pretrained_dir: str 
        ):  
        super().__init__()

        self.save_hyperparameters()


    def prepare_data(self):
        '''下载数据集和预训练模型.'''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, localpath=self.hparams.data_dir)
        oss.download_plm(self.hparams.plm, localpath=self.hparams.pretrained_dir)

    
    def setup(self, stage: str) -> None:
        """读取数据集, 对数据设置转换"""
        data = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([triple['predicate'] for triples in data['train']['triples'] for triple in triples]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams['label2id'] = label2id
        self.hparams['id2label'] = id2label
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.plm)
        self.bert_config = BertConfig.from_pretrained(self.hparams.pretrained_dir + self.hparams.plm)
        self.hparams['bert_config'] = self.bert_config
        self.hparams['token2id'] = dict(self.tokenizer.vocab)
        data.set_transform(transform=self.transform)
        self.train_dataset = data['train']
        self.valid_dataset = data['validation']
        if 'test' in data:
            self.test_dataset = data['test']


    def transform(self, example) -> Dict:
        batch_text = example['text']
        batch_triples = example['triples']
        batch_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'span_ids': [], 'head_ids': [], 'tail_ids': []}
        for i, text in enumerate(batch_text):
            tokens = fine_grade_tokenize(text, self.tokenizer)
            inputs = self.tokenizer.encode_plus(
                tokens, 
                is_pretokenized=True,
                padding='max_length',  
                max_length=self.hparams.max_length,
                add_special_tokens=True,
                truncation=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            triples = batch_triples[i]
            span_ids = torch.zeros(2, self.hparams.max_length, self.hparams.max_length)
            head_ids = torch.zeros(len(self.hparams.label2id), self.hparams.max_length, self.hparams.max_length)
            tail_ids = torch.zeros(len(self.hparams.label2id), self.hparams.max_length, self.hparams.max_length)
            for triple in triples:
                #加1是因为有cls
                span_ids[0][triple['subject_index'][0]+1][triple['subject_index'][1]+1] = 1
                span_ids[1][triple['object_index'][0]+1][triple['object_index'][1]+1] = 1
                head_ids[self.hparams.label2id[triple['predicate']]][triple['subject_index'][0]+1][triple['object_index'][0]+1] = 1
                tail_ids[self.hparams.label2id[triple['predicate']]][triple['subject_index'][1]+1][triple['object_index'][1]+1] = 1
            batch_inputs['span_ids'].append(span_ids)
            batch_inputs['head_ids'].append(head_ids)
            batch_inputs['tail_ids'].append(tail_ids)
        batch_inputs['span_ids'] = torch.stack(batch_inputs['span_ids'], dim=0)
        batch_inputs['head_ids'] = torch.stack(batch_inputs['head_ids'], dim=0)
        batch_inputs['tail_ids'] = torch.stack(batch_inputs['tail_ids'], dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        return batch


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