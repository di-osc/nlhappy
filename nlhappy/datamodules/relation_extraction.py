import pytorch_lightning as pl
from ..utils.make_datamodule import prepare_data_from_oss, char_idx_to_token
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import os
import logging
from functools import lru_cache

log = logging.getLogger(__name__)

class RelationExtractionDataModule(pl.LightningDataModule):
    """三元组抽取数据模块,
        数据集样式:
            {'text':'小明是小花的朋友'。
            'triples': [{'object': {'offset': [0, 2], 'text': '小明'},'predicate': '朋友','subject': {'offset': [3, 5], 'text': '小花'}}]}}

        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            transform(str): 转换特征格式,可以通过get_available_transforms查看
            max_length (int): 包括特殊token的最大序列长度,如果是None则自动设置最大长度,默认为None
            batch_size (int): 训练,验证,测试数据集的批次大小,
            num_workers (int): 多进程数
            pin_memory (bool): 是否应用锁页内存,
            dataset_dir (str): 数据集默认路径
            plm_dir (str): 预训练模型默认路径
        """
    def __init__(self, 
                 dataset: str, 
                 plm: str,
                 transform: str, 
                 batch_size: int ,
                 max_length: int = -1,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 dataset_dir: str ='datasets' ,
                 plm_dir: str = 'plms'):
        super().__init__()

        self.save_hyperparameters()
        self.transforms = {'gplinker': self.gplinker_transform,
                           'onerel': self.onerel_transform}
        assert self.hparams.transform in self.transforms.keys(), f'availabel models for relation extraction: {self.transforms.keys()}'


    def prepare_data(self):
        '''下载数据集和预训练模型.'''
        prepare_data_from_oss(dataset=self.hparams.dataset,
                              plm=self.hparams.plm,
                              dataset_dir=self.hparams.dataset_dir,
                              plm_dir=self.hparams.plm_dir)

    
    def setup(self, stage: str) -> None:
        """对数据设置转换"""
        self.hparams.trf_config = self.trf_config
        self.hparams.label2id = self.label2id
        self.hparams.id2label = {i:l for l,i in self.label2id.items()}
        if self.hparams.transform == 'onerel':
            self.hparams.tag2id = self.onerel_tag2id
        self.hparams.vocab = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        self.hparams.max_length = self.max_length
        # 设置数据转换
        self.dataset.set_transform(transform=self.transforms.get(self.hparams.transform))
        
    
    @property
    @lru_cache()
    def dataset(self):
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        try:
            ds = load_from_disk(dataset_path)
            return ds
        except:
            print(f'load dataset failed from {dataset_path}')
            
    
    @property
    @lru_cache()
    def max_length(self):
        if self.hparams.max_length == -1:
            return min(512, max([len(d['text'])+2 for d in self.dataset['train']]))
        else:
            return self.hparams.max_length
        
    
    @property
    @lru_cache()
    def trf_config(self):
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        return AutoConfig.from_pretrained(plm_path)
    
    
    @property
    @lru_cache()
    def tokenizer(self):
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        return AutoTokenizer.from_pretrained(plm_path)
        
        
    @property
    @lru_cache   
    def label2id(self) -> Dict:
        labels = sorted(set([triple['predicate'] for triples in self.dataset['train']['triples'] for triple in triples]))
        label2id = {label: i for i, label in enumerate(labels)}
        return label2id
    
    
    @property
    def onerel_tag2id(self):
        return {'HB-TB':1, 'HB-TE':2, 'HE-TE':3, 'O':0}
    
    
    @classmethod
    def get_availabel_transforms(cls)-> List[str]:
        return ['gplinker', 'onerel']
    


    def train_dataloader(self):
        return DataLoader(dataset=self.dataset['train'], 
                          batch_size=self.hparams.batch_size, 
                          shuffle=True,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers)


    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['validation'],
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          pin_memory=self.hparams.pin_memory,
                          num_workers=self.hparams.num_workers)        


    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'],
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            pin_memory=self.hparams.pin_memory,
                            num_workers=self.hparams.num_workers)
        
        
        
    def gplinker_transform(self, example) -> Dict:
        batch_text = example['text']
        batch_triples = example['triples']
        batch_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'so_ids': [], 'head_ids': [], 'tail_ids': []}
        for i, text in enumerate(batch_text):
            inputs = self.tokenizer(
                text, 
                padding='max_length',  
                max_length=self.hparams.max_length,
                truncation=True,
                return_offsets_mapping=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            offset_mapping = inputs['offset_mapping']
            triples = batch_triples[i]
            so_ids = torch.zeros(2, self.hparams.max_length, self.hparams.max_length)
            # span_ids = torch.zeros(len(self.hparams['s_label2id']), self.hparams.max_length, self.hparams.max_length)
            head_ids = torch.zeros(len(self.hparams['label2id']), self.hparams.max_length, self.hparams.max_length)
            tail_ids = torch.zeros(len(self.hparams['label2id']), self.hparams.max_length, self.hparams.max_length)
            for triple in triples:
                try:
                    sub_start = triple['subject']['offset'][0]
                    sub_end = triple['subject']['offset'][1]-1
                    sub_start = char_idx_to_token(sub_start, offset_mapping=offset_mapping)
                    sub_end = char_idx_to_token(sub_end, offset_mapping=offset_mapping)
                    obj_start = triple['object']['offset'][0]
                    obj_end = triple['object']['offset'][1]-1
                    obj_start = char_idx_to_token(obj_start, offset_mapping=offset_mapping)
                    obj_end = char_idx_to_token(obj_end, offset_mapping=offset_mapping)
                    so_ids[0][sub_start][sub_end] = 1
                    so_ids[1][obj_start][obj_end] = 1
                    head_ids[self.hparams['label2id'][triple['predicate']]][sub_start][obj_start] = 1
                    tail_ids[self.hparams['label2id'][triple['predicate']]][sub_end][obj_end] = 1
                except:
                    log.warning(f'sub char offset {(sub_start, sub_end)} or obj char offset {(obj_start, obj_end)} align to token offset failed in \n{text}')
                    pass
            batch_inputs['so_ids'].append(so_ids)
            batch_inputs['head_ids'].append(head_ids)
            batch_inputs['tail_ids'].append(tail_ids)
        batch_inputs['so_ids'] = torch.stack(batch_inputs['so_ids'], dim=0)
        batch_inputs['head_ids'] = torch.stack(batch_inputs['head_ids'], dim=0)
        batch_inputs['tail_ids'] = torch.stack(batch_inputs['tail_ids'], dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        return batch
    
    
    def onerel_transform(self, example):
        texts = example['text']
        batch_triples = example['triples']
        batch_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'tag_ids': []}
        for i, text in enumerate(texts):
            inputs = self.tokenizer(text, 
                                    padding='max_length',  
                                    max_length=self.hparams.max_length,
                                    truncation=True,
                                    return_offsets_mapping=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            offset_mapping = inputs['offset_mapping']
            triples = batch_triples[i]
            tag_ids = torch.zeros(len(self.label2id), self.hparams.max_length, self.hparams.max_length)
            for triple in triples:
                try:
                    rel = triple['predicate']
                    rel_id = self.label2id[rel]
                    sub_start = triple['subject']['offset'][0]
                    sub_end = triple['subject']['offset'][1]-1
                    sub_start = char_idx_to_token(sub_start, offset_mapping=offset_mapping)
                    sub_end = char_idx_to_token(sub_end, offset_mapping=offset_mapping)
                    obj_start = triple['object']['offset'][0]
                    obj_end = triple['object']['offset'][1]-1
                    obj_start = char_idx_to_token(obj_start, offset_mapping=offset_mapping)
                    obj_end = char_idx_to_token(obj_end, offset_mapping=offset_mapping)
                    tag_ids[rel_id][sub_start][obj_start] = self.onerel_tag2id['HB-TB']
                    tag_ids[rel_id][sub_start][obj_end] = self.onerel_tag2id['HB-TE']
                    tag_ids[rel_id][sub_end][obj_end] = self.onerel_tag2id['HE-TE']
                except Exception as e:
                    log.exception(e)
                    log.warning(f'sub char offset {(sub_start, sub_end)} or obj char offset {(obj_start, obj_end)} align to token offset failed in \n{text}')
                    pass
            batch_inputs['tag_ids'].append(tag_ids)
        batch_inputs['tag_ids'] = torch.stack(batch_inputs['tag_ids'], dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        return batch
                    
                
