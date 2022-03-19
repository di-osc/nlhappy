import pytorch_lightning as pl
from typing import Optional,  Tuple, List, Dict
from transformers import BertTokenizer, BertConfig
from ..utils.storer import OSSStorer
import pandas as pd
from ..utils import utils
from torch.utils.data import DataLoader
import os
from ..utils.preprocessing import fine_grade_tokenize
import zipfile
from datasets import load_from_disk
import torch

class TextClassificationDataModule(pl.LightningDataModule):
    '''
    文本分类
    '''
    def __init__(
        self,
        dataset: str,
        plm: str,
        max_length: int ,
        batch_size: int ,
        is_multi_label: bool=False,
        num_workers: int =1,
        pin_memory: bool =True,
        pretrained_dir: str ='./plms/',
        data_dir: str ='./datasets/',
        label_pad_id: int =-100, 
        ):  
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)
        
        


    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.data_dir)
        oss.download_plm(self.hparams.plm, self.hparams.pretrained_dir)
        

    def transform(self, examples) -> Dict:
        batch_text = examples['text']
        batch_inputs = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[], 'label_ids': []}
        for i, text in enumerate(batch_text):
            tokens = fine_grade_tokenize(text, self.tokenizer)
            inputs = self.tokenizer.encode_plus(
                tokens, 
                padding='max_length',  
                max_length=self.hparams.max_length,
                add_special_tokens=True,
                truncation=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            if not self.hparams.is_multi_label:
                batch_inputs['label_ids'].append(torch.tensor(self.hparams['label2id'][examples['labels'][i][0]]))
            if self.hparams.is_multi_label:
                label_ids = torch.zeros(len(self.hparams.label2id))
                for label in examples['labels'][i]:
                    label_ids[self.hparams.label2id[label]] = 1
                batch_inputs['label_ids'].append(label_ids)
        batch_inputs['label_ids'] = torch.stack(batch_inputs['label_ids'], dim=0)
        batch_inputs = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        return batch_inputs
        

        
    def setup(self, stage: str) -> None:
        self.dataset = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([label for labels in self.dataset['train']['labels'] for label in labels]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams['label2id'] = label2id
        self.hparams['id2label'] = id2label
        self.dataset.set_transform(transform=self.transform)
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.plm)
        self.hparams['token2id'] = dict(self.tokenizer.vocab)
        bert_config = BertConfig.from_pretrained(self.hparams.pretrained_dir + self.hparams.plm)
        self.hparams['bert_config'] = bert_config
            

    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(
            dataset= self.dataset['train'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.dataset['validation'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.dataset['test'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)



if __name__ == "__main__":
    dm = TextClassificationDataModule(
        dataset='CHIP-CTC',
        plm='chinese-roberta-wwm-ext',
        max_length=512,
        is_multi_label=False,
        batch_size=2,
    )