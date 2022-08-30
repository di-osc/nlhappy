from functools import lru_cache
from ..utils.make_datamodule import char_idx_to_token, get_logger, PLMBaseDataModule
import torch
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import random


log = get_logger()


class RelationExtractionDataModule(PLMBaseDataModule):
    """三元组抽取数据模块,
        数据集样式:
            {'text':'小明是小花的朋友'。
            'triples': [{'object': {'offset': [0, 2], 'text': '小明'},'predicate': '朋友','subject': {'offset': [3, 5], 'text': '小花'}}]}}
        """
    def __init__(self, 
                 dataset: str, 
                 plm: str,
                 transform: str, 
                 batch_size: int ,
                 auto_length: Union[int, str] = 'max',
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 dataset_dir: str ='datasets',
                 plm_dir: str = 'plms'):
        """
        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            transform(str): 转换特征格式,可以通过get_available_transforms查看
            auto_length (str, int): 自动设置最大长度的策略, 可以为'max', 'mean'或者>0的数,超过512的设置为512
            batch_size (int): 训练,验证,测试数据集的批次大小,
            num_workers (int): 多进程数
            pin_memory (bool): 是否应用锁页内存,
            dataset_dir (str): 数据集默认路径
            plm_dir (str): 预训练模型默认路径
        """
        super().__init__()
                
        self.transforms = {'gplinker': self.gplinker_transform,
                           'onerel': self.onerel_transform,
                           'casrel': self.casrel_transform}
        assert self.hparams.transform in self.transforms.keys(), f'availabel models for relation extraction: {self.transforms.keys()}'


    def setup(self, stage: str) -> None:
        """对数据设置转换"""
        self.hparams.max_length = self.get_max_length()
        if self.hparams.transform == 'onerel':
            self.hparams.tag2id = self.onerel_tag2id
            self.hparams.id2tag = {i:l for l,i in self.onerel_tag2id.items()}
        self.hparams.label2id = self.label2id
        self.hparams.id2label = {i:l for l, i in self.label2id.items()}
        # 设置数据转换
        self.dataset.set_transform(transform=self.transforms.get(self.hparams.transform))   

    
    @property
    @lru_cache
    def label2id(self):
        labels = pd.Series(np.concatenate(self.train_df.triples.values)).apply(lambda x: x['predicate']).drop_duplicates().values
        label2id = {label: i for i, label in enumerate(labels)}
        return label2id
    
        
    def gplinker_transform(self, example) -> Dict:
        batch_text = example['text']
        batch_triples = example['triples']
        batch_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        batch_so_ids = []
        batch_head_ids = []
        batch_tail_ids = []
        for i, text in enumerate(batch_text):
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
            so_ids = torch.zeros(2, self.hparams.max_length, self.hparams.max_length)
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
            batch_so_ids.append(so_ids)
            batch_head_ids.append(head_ids)
            batch_tail_ids.append(tail_ids)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        batch['so_ids'] = torch.stack(batch_so_ids, dim=0)
        batch['head_ids'] = torch.stack(batch_head_ids, dim=0)
        batch['tail_ids'] = torch.stack(batch_tail_ids, dim=0)
        return batch
    
    
    @property
    def onerel_tag2id(self):
        return {'O':0, 'HB-TB':1, 'HB-TE':2, 'HE-TE':3}
    
    
    def onerel_transform(self, example):
        texts = example['text']
        batch_triples = example['triples']
        batch_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        batch_tag_ids = []
        batch_loss_mask = []
        max_length = self.hparams.max_length
        for i, text in enumerate(texts):
            inputs = self.tokenizer(text, 
                                    padding='max_length',  
                                    max_length=max_length,
                                    truncation=True,
                                    return_offsets_mapping=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            offset_mapping = inputs['offset_mapping']
            triples = batch_triples[i]
            tag_ids = torch.zeros(len(self.label2id), max_length, max_length, dtype=torch.long)
            loss_mask = torch.ones(1, max_length, max_length, dtype=torch.long)
            att_mask = torch.tensor(inputs['attention_mask'])
            loss_mask = loss_mask * att_mask.unsqueeze(0) * att_mask.unsqueeze(0).T
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
                    if sub_start != sub_end and obj_start != obj_end:
                        tag_ids[rel_id][sub_start][obj_start] = self.onerel_tag2id['HB-TB']
                        tag_ids[rel_id][sub_start][obj_end] = self.onerel_tag2id['HB-TE']
                        tag_ids[rel_id][sub_end][obj_end] = self.onerel_tag2id['HE-TE']
                    if sub_start == sub_end and obj_start != obj_end:
                        tag_ids[rel_id][sub_start][obj_start] = self.onerel_tag2id['HB-TB']
                        tag_ids[rel_id][sub_end][obj_end] = self.onerel_tag2id['HE-TE']
                    if sub_start != sub_end and obj_start == obj_end:
                        tag_ids[rel_id][sub_start][obj_start] = self.onerel_tag2id['HB-TB']
                        tag_ids[rel_id][sub_end][obj_end] = self.onerel_tag2id['HE-TE']
                    if sub_start == sub_end and obj_start == obj_end:
                        tag_ids[rel_id][sub_start][obj_start] = self.onerel_tag2id['HB-TB']
                except Exception as e:
                    log.exception(e)
                    log.warning(f'sub char offset {(sub_start, sub_end)} or obj char offset {(obj_start, obj_end)} align to token offset failed in \n{text}')
                    pass
            batch_tag_ids.append(tag_ids)
            batch_loss_mask.append(loss_mask)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        batch['tag_ids'] = torch.stack(batch_tag_ids, dim=0)
        batch['loss_mask'] = torch.stack(batch_loss_mask, dim=0)
        return batch
    
    
    def casrel_transform(self, example):
        texts = example['text']
        batch_triples = example['triples']
        batch_inputs = {'input_ids': [], 
                        'attention_mask': [], 
                        'token_type_ids': []}
        batch_subs = []
        batch_objs = []
        batch_sub = []
                        
        max_length = self.hparams.max_length
        for i, text in enumerate(texts):
            inputs = self.tokenizer(text, 
                                    padding='max_length',  
                                    max_length=max_length,
                                    truncation=True,
                                    return_offsets_mapping=True)
            offset_mapping = inputs['offset_mapping']
            triples = batch_triples[i]
            s2ro_map = {}
            for triple in triples:
                sub_start = triple['subject']['offset'][0]
                sub_end = triple['subject']['offset'][1]-1
                sub_start = char_idx_to_token(sub_start, offset_mapping=offset_mapping)
                sub_end = char_idx_to_token(sub_end, offset_mapping=offset_mapping)
                obj_start = triple['object']['offset'][0]
                obj_end = triple['object']['offset'][1]-1
                obj_start = char_idx_to_token(obj_start, offset_mapping=offset_mapping)
                obj_end = char_idx_to_token(obj_end, offset_mapping=offset_mapping)
                rel_id = self.hparams.label2id[triple['predicate']]
                if (sub_start, sub_end) not in s2ro_map:
                    s2ro_map[(sub_start, sub_end)] = [] 
                s2ro_map[(sub_start, sub_end)].append((obj_start, obj_end, rel_id))
                
            if s2ro_map:
                subs = torch.zeros(max_length, 2, dtype=torch.float)
                for s in s2ro_map:
                    subs[s[0], 0] = 1
                    subs[s[1], 1] = 1
                # for sub_head_idx, sub_tail_idx in list(s2ro_map.keys()):
                sub_head_idx, sub_tail_idx = random.choice(list(s2ro_map.keys()))
                sub = torch.tensor([sub_head_idx, sub_tail_idx], dtype=torch.long)
                objs = torch.zeros((max_length, len(self.hparams.label2id), 2), dtype=torch.float)
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    objs[ro[0], ro[2], 0] = 1
                    objs[ro[1], ro[2], 1] = 1
                batch_subs.append(subs)
                batch_objs.append(objs)
                batch_sub.append(sub)
                batch_inputs['input_ids'].append(inputs['input_ids'])
                batch_inputs['attention_mask'].append(inputs['attention_mask'])
                batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        batch['subs'] = torch.stack(batch_subs, dim=0)
        batch['sub'] = torch.stack(batch_sub, dim=0)
        batch['objs'] = torch.stack(batch_objs, dim=0)
        return batch
        

    
    @staticmethod
    def get_one_example():
        return {'text':'小明是小花的朋友',
                'triples': [{'object': {'offset': [0, 2], 'text': '小明'},
                             'predicate': '朋友',
                             'subject': {'offset': [3, 5], 'text': '小花'}}]}
                
                    
                
