from functools import lru_cache
from ..utils.make_datamodule import char_idx_to_token, get_logger, PLMBaseDataModule, sequence_padding
import torch
from typing import Dict, Union, List
import numpy as np
import pandas as pd
import random


log = get_logger()


class RelationExtractionDataModule(PLMBaseDataModule):
    """三元组抽取数据模块,
        数据集样式:
            {'text':'小明是小花的朋友'。
            'rels': [{'o': {'indices': [0, 1, 2], 'text': '小明', 'label': '人物'},'p': '朋友','s': {'indices': [3, 4, 5], 'text': '小花', 'label': '人物'}}]}}
        数据集说明:
            当只抽取三元组的时候subject,object中的标签可以为''
        """
    def __init__(self, 
                 dataset: str, 
                 batch_size: int ,
                 plm: str = 'hfl/chinese-roberta-wwm-ext',
                 **kwargs):
        """
        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            auto_length (str, int): 自动设置最大长度的策略, 可以为'max', 'mean'或者>0的数,超过512的设置为512
            batch_size (int): 训练,验证,测试数据集的批次大小,
            num_workers (int): 多进程数
            pin_memory (bool): 是否应用锁页内存,
            dataset_dir (str): 数据集默认路径
            plm_dir (str): 预训练模型默认路径
        """
        super().__init__()


    def setup(self, stage: str = 'fit') -> None:
        """对数据设置转换"""
        self.hparams.id2rel = self.id2rel
        self.hparams.id2combined = self.id2combined
        self.hparams.id2onerel = self.id2onerel
        self.hparams.id2ent = self.id2ent
        
    @property
    @lru_cache()
    def combined_labels(self) -> List:
        """将主体客体与其实体标签结合起来, (主体, 地点)
        """
        def get_labels(e):
            return ['主体'+'-'+e['s']['label'], '客体'+'-'+ e['o']['label']]
        labels = set(np.concatenate(pd.Series(np.concatenate(self.train_df.rels)).apply(get_labels)))
        ls = [tuple(l.split('-')) for l in labels]
        return ls
    
    @property
    @lru_cache()
    def combined2id(self) -> Dict:
        return {l:i for i,l in enumerate(self.combined_labels)}
    
    @property
    def id2combined(self) -> Dict:
        return {i:l for l,i in self.combined2id.items()}
    
    @property
    @lru_cache()
    def ent_labels(self) -> List:
        def get_labels(e):
            return [e['s']['label'], e['o']['label']]
        return list(set(np.concatenate(pd.Series(np.concatenate(self.train_df.rels)).apply(get_labels))))
    
    @property
    def ent2id(self) -> Dict:
        return {l:i for i,l in enumerate(self.ent_labels)}
    
    @property
    def id2ent(self) -> Dict:
        return {i:l for l,i in self.ent2id.items()}
    
    
    @property
    @lru_cache()
    def rel_labels(self) -> List:
        labels = set(pd.Series(np.concatenate(self.train_df.rels.values)).apply(lambda x: x['p']))
        return list(labels)

    @property
    def rel2id(self) -> Dict:
        label2id = {label: i for i, label in enumerate(self.rel_labels)}
        return label2id
    
    @property
    def id2rel(self) -> Dict:
        return {i:l for l,i in self.rel2id.items()}
    
    @property
    def onerel2id(self):
        return {'O':0, 'HB-TB':1, 'HB-TE':2, 'HE-TE':3}
    
    @property
    def id2onerel(self) -> Dict:
        return {i:l for l,i in self.onerel2id.items()}
    
    def sparse_triple_transform(self, examples):
        batch_text = examples['text']
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_inputs = self.tokenizer(batch_text,
                                      padding=True,
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors='pt',
                                      return_token_type_ids=False)
        batch_so_tags = []
        batch_head_tags = []
        batch_tail_tags = []
        batch_triples = examples['rels']
        for i, text in enumerate(batch_text):
            triples = batch_triples[i]
            so_tags = [set() for _ in range(2)]
            head_tags = [set() for _ in range(len(self.rel_labels))]
            tail_tags = [set() for _ in range(len(self.rel_labels))]
            for triple in triples:
                try:
                    sub_head = triple['s']['indices'][0]
                    _sub_head = batch_inputs.char_to_token(i, sub_head)
                    sub_tail = triple['s']['indices'][-1]
                    _sub_tail = batch_inputs.char_to_token(i, sub_tail)
                    sub_label = triple['s']['label']
                    assert _sub_head is not None and _sub_tail is not None
                except:
                    log.warning(f'subject {(sub_head, sub_tail)} align fail in \n {text}')
                    continue
                
                try:
                    obj_head = triple['o']['indices'][0]
                    obj_tail = triple['o']['indices'][-1]
                    obj_label = triple['o']['label']
                    _obj_head = batch_inputs.char_to_token(i, obj_head)
                    _obj_tail = batch_inputs.char_to_token(i, obj_tail)
                    assert _obj_head is not None and _obj_tail is not None
                except:
                    log.warning(f'object {(obj_head, obj_tail)} align fail in \n {text}')
                    continue
                
                so_tags[0].add((_sub_head, _sub_tail))
                so_tags[1].add((_obj_head, _obj_tail))
                head_tags[self.rel2id[triple['p']]].add((_sub_head, _obj_head))
                tail_tags[self.rel2id[triple['p']]].add((_sub_tail, _obj_tail))
            for tag in so_tags + head_tags + tail_tags:
                if not tag:
                    tag.add((0,0))
            so_tags = sequence_padding([list(l) for l in so_tags])
            head_tags = sequence_padding([list(l) for l in head_tags])
            tail_tags = sequence_padding([list(l) for l in tail_tags])
            batch_so_tags.append(so_tags)
            batch_head_tags.append(head_tags)
            batch_tail_tags.append(tail_tags)
        batch_inputs['so_tags'] = torch.tensor(sequence_padding(batch_so_tags, seq_dims=2), dtype=torch.long)
        batch_inputs['head_tags'] = torch.tensor(sequence_padding(batch_head_tags, seq_dims=2), dtype=torch.long)
        batch_inputs['tail_tags'] = torch.tensor(sequence_padding(batch_tail_tags, seq_dims=2), dtype=torch.long)
        return batch_inputs
    
        
    def triple_transform(self, example) -> Dict:
        batch_text = example['text']
        batch_triples = example['rels']
        batch_so_ids = []
        batch_head_ids = []
        batch_tail_ids = []
        max_length = self.get_batch_max_length(batch_text)
        batch_inputs = self.tokenizer(batch_text, 
                                      max_length=max_length,
                                      padding=True,
                                      truncation=True,
                                      return_token_type_ids=False,
                                      return_tensors='pt')
        for i, text in enumerate(batch_text):
            triples = batch_triples[i]
            so_ids = torch.zeros(2, max_length, max_length, dtype=torch.long)
            head_ids = torch.zeros(len(self.rel_labels), max_length, max_length, dtype=torch.long)
            tail_ids = torch.zeros(len(self.rel_labels), max_length, max_length, dtype=torch.long)
            for triple in triples:
                try:
                    sub_start = triple['s']['indices'][0]
                    sub_end = triple['s']['indices'][-1]
                    _sub_start = batch_inputs.char_to_token(i, sub_start)
                    _sub_end = batch_inputs.char_to_token(i, sub_end)
                    obj_start = triple['o']['indices'][0]
                    obj_end = triple['o']['indices'][-1]
                    _obj_start = batch_inputs.char_to_token(i, obj_start)
                    _obj_end = batch_inputs.char_to_token(i, obj_end)
                    so_ids[0][_sub_start][_sub_end] = 1
                    so_ids[1][_obj_start][_obj_end] = 1
                    head_ids[self.rel2id[triple['p']]][_sub_start][_obj_start] = 1
                    tail_ids[self.rel2id[triple['p']]][_sub_end][_obj_end] = 1
                except:
                    log.warning(f'sub char offset {(sub_start, sub_end)} or obj char offset {(obj_start, obj_end)} align to token offset failed in \n{text}')
                    pass
            batch_so_ids.append(so_ids)
            batch_head_ids.append(head_ids)
            batch_tail_ids.append(tail_ids)
        batch_inputs['so_tags'] = torch.stack(batch_so_ids, dim=0)
        batch_inputs['head_tags'] = torch.stack(batch_head_ids, dim=0)
        batch_inputs['tail_tags'] = torch.stack(batch_tail_ids, dim=0)
        return batch_inputs
    
    
    def onerel_transform(self, example):
        texts = example['text']
        batch_triples = example['rels']
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
                    rel = triple['p']
                    rel_id = self.label2id[rel]
                    sub_start = triple['s']['indices'][0]
                    sub_end = triple['s']['indices'][-1]
                    sub_start = char_idx_to_token(sub_start, offset_mapping=offset_mapping)
                    sub_end = char_idx_to_token(sub_end, offset_mapping=offset_mapping)
                    obj_start = triple['o']['indices'][0]
                    obj_end = triple['o']['indices'][-1]
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
        batch_triples = example['rels']
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
                sub_start = triple['s']['indices'][0]
                sub_end = triple['s']['indices'][-1]
                sub_start = char_idx_to_token(sub_start, offset_mapping=offset_mapping)
                sub_end = char_idx_to_token(sub_end, offset_mapping=offset_mapping)
                obj_start = triple['o']['indices'][0]
                obj_end = triple['o']['indices'][-1]
                obj_start = char_idx_to_token(obj_start, offset_mapping=offset_mapping)
                obj_end = char_idx_to_token(obj_end, offset_mapping=offset_mapping)
                rel_id = self.hparams.label2id[triple['p']]
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
        

    def sparse_combined_transform(self, examples):
        batch_text = examples['text']
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_inputs = self.tokenizer(batch_text,
                                      padding=True,
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors='pt',
                                      return_token_type_ids=False)
        batch_combined_tags = []
        batch_head_tags = []
        batch_tail_tags = []
        batch_triples = examples['rels']
        for i, text in enumerate(batch_text):
            triples = batch_triples[i]
            combined_tags = [set() for _ in range(len(self.combined_labels))]
            head_tags = [set() for _ in range(len(self.rel_labels))]
            tail_tags = [set() for _ in range(len(self.rel_labels))]
            for triple in triples:
                try:
                    sub_head = triple['s']['indices'][0]
                    _sub_head = batch_inputs.char_to_token(i, sub_head)
                    sub_tail = triple['s']['indices'][-1]
                    _sub_tail = batch_inputs.char_to_token(i, sub_tail)
                    sub_label = triple['s']['label']
                    assert _sub_head is not None and _sub_tail is not None
                except:
                    log.warning(f'subject {(sub_head, sub_tail)} align fail in \n {text}')
                    continue
                
                try:
                    obj_head = triple['o']['indices'][0]
                    obj_tail = triple['o']['indices'][-1]
                    obj_label = triple['o']['label']
                    _obj_head = batch_inputs.char_to_token(i, obj_head)
                    _obj_tail = batch_inputs.char_to_token(i, obj_tail)
                    assert _obj_head is not None and _obj_tail is not None
                except:
                    log.warning(f'object {(obj_head, obj_tail)} align fail in \n {text}')
                    continue
                
                combined_tags[self.combined2id[('主体', sub_label)]].add((_sub_head, _sub_tail))
                combined_tags[self.combined2id[('客体', obj_label)]].add((_obj_head, _obj_tail))
                head_tags[self.rel2id[triple['p']]].add((_sub_head, _obj_head))
                tail_tags[self.rel2id[triple['p']]].add((_sub_tail, _obj_tail))
            for tag in combined_tags + head_tags + tail_tags:
                if not tag:
                    tag.add((0,0))
            combined_tags = sequence_padding([list(l) for l in combined_tags])
            head_tags = sequence_padding([list(l) for l in head_tags])
            tail_tags = sequence_padding([list(l) for l in tail_tags])
            batch_combined_tags.append(combined_tags)
            batch_head_tags.append(head_tags)
            batch_tail_tags.append(tail_tags)
        batch_inputs['combined_tags'] = torch.tensor(sequence_padding(batch_combined_tags, seq_dims=2), dtype=torch.long)
        batch_inputs['head_tags'] = torch.tensor(sequence_padding(batch_head_tags, seq_dims=2), dtype=torch.long)
        batch_inputs['tail_tags'] = torch.tensor(sequence_padding(batch_tail_tags, seq_dims=2), dtype=torch.long)
        return batch_inputs
    
    
    def combined_transform(self, examples):
        batch_text = examples['text']
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_inputs = self.tokenizer(batch_text,
                                      padding=True,
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors='pt',
                                      return_token_type_ids=False)
        batch_combined_tags = []
        batch_head_tags = []
        batch_tail_tags = []
        batch_triples = examples['rels']
        for i, text in enumerate(batch_text):
            triples = batch_triples[i]
            combined_tags = torch.zeros(len(self.combined_labels), max_length, max_length, dtype=torch.long)
            head_tags = torch.zeros(len(self.rel_labels), max_length, max_length, dtype=torch.long)
            tail_tags = torch.zeros(len(self.rel_labels), max_length, max_length, dtype=torch.long)
            for triple in triples:
                sub_head = triple['s']['indices'][0]
                _sub_head = batch_inputs.char_to_token(i, sub_head)
                sub_tail = triple['s']['indices'][-1]
                _sub_tail = batch_inputs.char_to_token(i, sub_tail)
                sub_label = triple['s']['label']
                
                obj_head = triple['o']['indices'][0]
                obj_tail = triple['o']['indices'][-1]
                obj_label = triple['o']['label']
                _obj_head = batch_inputs.char_to_token(i, obj_head)
                _obj_tail = batch_inputs.char_to_token(i, obj_tail)
                
                if _sub_head is None or _sub_tail is None :
                    log.warning(f'subject {(sub_head, sub_tail)} align fail in \n {text}')
                    continue
                if _obj_head is None or _obj_tail is None:
                    log.warning(f'object {(obj_head, obj_tail)} align fail in \n {text}')
                    continue
                
                combined_tags[self.combined2id[('主体', sub_label)], _sub_head, _sub_tail] = 1
                combined_tags[self.combined2id[('客体', obj_label)], _obj_head, _obj_tail] = 1
                head_tags[self.rel2id[triple['p']], _sub_head, _obj_head] = 1
                tail_tags[self.rel2id[triple['p']], _sub_tail, _obj_tail] = 1
            batch_combined_tags.append(combined_tags)
            batch_head_tags.append(head_tags)
            batch_tail_tags.append(tail_tags)
        batch_inputs['combined_tags'] = torch.stack(batch_combined_tags, dim=0)
        batch_inputs['head_tags'] = torch.stack(batch_head_tags, dim=0)
        batch_inputs['tail_tags'] = torch.stack(batch_tail_tags, dim=0)
        return batch_inputs