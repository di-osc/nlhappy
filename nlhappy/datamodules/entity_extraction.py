from functools import lru_cache
from ..utils.make_datamodule import PLMBaseDataModule, char_idx_to_token
from ..utils.utils import get_logger
import pandas as pd
import numpy as np
import torch
from typing import List


log = get_logger()

class EntityExtractionDataModule(PLMBaseDataModule):
    """实体抽取数据模块
    可以解决嵌套实体和非连续实体
    globalpointer: 可以抽取嵌套实体,不能抽取非连续实体
    w2ner: 可以抽取非连续和嵌套实体
    
    数据集格式: {"text":"这是一个长颈鹿","ents":[{"indices":[4,5,6],"label":"动物", "text":"长颈鹿"}]}
    """
        
    def __init__(self,
                 dataset: str,
                 batch_size: int,
                 transform: str = 't2',
                 plm: str = 'hfl/chinese-roberta-wwm-ext',
                 **kwargs):
        super().__init__()
        self.transforms['w2'] = self.w2ner_transform
        self.transforms['t2'] = self.tp_transform
        self.transforms['bio'] = self.bio_transform
        
        assert self.hparams.transform in self.transforms.keys(), f'availabel transforms {list(self.transforms.keys())}'

    
    def setup(self, stage: str = 'fit'):
        self.hparams.max_length = self.get_max_length()
        if self.hparams.transform == 'w2':
            labels = list(self.ent2id.keys())
            label2id = {'<pad>':0, '<suc>':1}
            for i, label in enumerate(labels):
                label2id[label] = i+2
            id2label = {i:l for l,i in label2id.items()}
            self.hparams.label2id = label2id
            self.hparams.id2label = id2label
        elif self.hparams.transform == 'bio':
            self.hparams.label2id = self.bio2id
            
        else :
            self.hparams.label2id = self.ent2id
            self.hparams.id2label = self.id2ent
        self.dataset.set_transform(self.transforms.get(self.hparams.transform)) 
    
    
    @classmethod
    def get_available_transforms(cls):
        return ['w2', 't2', 'bio']
    
    
    @property
    @lru_cache()
    def ent_labels(self) -> List:
        labels = sorted(pd.Series(np.concatenate(self.train_df['ents'])).apply(lambda x: x['label']).drop_duplicates().values)
        return labels
    
    
    @property
    @lru_cache()
    def bio_labels(self) -> List:
        b_labels = ['B' + '-' + l for l in self.ent_labels]
        i_labels = ['I' + '-' + l for l in self.ent_labels]
        return ['O' + b_labels + i_labels]
    
    @property
    def bio2id(self):
        return {l:i for i,l in enumerate(self.bio_labels)}


    @property
    def ent2id(self):
        return {l:i for i,l in enumerate(self.ent_labels)}


    @property
    def id2ent(self):
        return {i:l for l,i in self.ent2id.items()}


    @lru_cache()
    def get_dis2idx(self):
        dis2idx = np.zeros((1000), dtype='int64')
        dis2idx[1] = 1
        dis2idx[2:] = 2
        dis2idx[4:] = 3
        dis2idx[8:] = 4
        dis2idx[16:] = 5
        dis2idx[32:] = 6
        dis2idx[64:] = 7
        dis2idx[128:] = 8
        dis2idx[256:] = 9
        return dis2idx


    def fill(self, data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = np.array(x, dtype=np.int32)
        return new_data


    def w2ner_transform(self, examples):
        max_length = self.hparams.max_length
        batch_text = examples['text']
        batch_ents = examples['ents']
        batch_inputs = self.tokenizer(batch_text,
                                      max_length=max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_offsets_mapping=True,
                                      add_special_tokens=False,
                                      return_tensors='pt')
        batch_mappings = batch_inputs.pop('offset_mapping').tolist()
        batch_lengths = batch_inputs['attention_mask'].sum(dim=-1).tolist()
        batch_size = len(batch_text)
        batch_dist_ids = np.zeros((batch_size, max_length, max_length), dtype=np.int)
        batch_dist = []
        batch_label_ids = []
        dis2idx = self.get_dis2idx()
        for i, text in enumerate(batch_text):
            mapping = batch_mappings[i]
            ents = batch_ents[i]
            length = batch_lengths[i]
            label_ids = np.zeros((max_length, max_length), dtype=np.int)
            for ent in ents:
                idx_ls = ent['indices']
                for i in range(len(idx_ls)):
                    if i +1 >= len(idx_ls):
                        break
                    label_ids[char_idx_to_token(idx_ls[i], mapping), char_idx_to_token(idx_ls[i+1], mapping)] = 1
                label_ids[char_idx_to_token(idx_ls[-1], mapping), char_idx_to_token(idx_ls[0], mapping)] = self.label2id[ent['label']]
            batch_label_ids.append(label_ids)

            _dist_inputs = np.zeros((length, length), dtype=np.int)
            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i in range(length):
                for j in range(length):
                    if _dist_inputs[i, j] < 0:
                        _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                    else:
                        _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
            _dist_inputs[_dist_inputs == 0] = 19
            batch_dist.append(_dist_inputs)

        
        batch_label_ids = np.stack(batch_label_ids)
        batch_inputs['label_ids'] = torch.from_numpy(batch_label_ids)
        _batch_dist_ids = self.fill(batch_dist, batch_dist_ids)
        batch_dist_ids = np.stack(_batch_dist_ids)
        batch_inputs['distance_ids'] = torch.from_numpy(batch_dist_ids)
        return batch_inputs


    def tp_transform(self, examples):
        batch_text = examples['text']
        batch_ents = examples['ents']
        max_length = self.get_max_length()
        batch_inputs = self.tokenizer(batch_text,
                                      max_length=max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_offsets_mapping=True,
                                      add_special_tokens=False,
                                      return_tensors='pt')
        batch_mappings = batch_inputs.pop('offset_mapping').tolist()
        batch_label_ids = []
        for i, text in enumerate(batch_text):
            ents = batch_ents[i]
            mapping = batch_mappings[i]
            label_ids = torch.zeros(len(self.hparams.label2id), max_length, max_length)
            for ent in ents :
                if len(ent['indices']) == 0:
                    log.warn(f'found empty entity indexes in {text}')
                    continue
                start = ent['indices'][0]
                start = char_idx_to_token(start, mapping)
                end = ent['indices'][-1] 
                end = char_idx_to_token(end, mapping)
                label_id = self.hparams.label2id[ent['label']]
                label_ids[label_id,  start, end] = 1
            batch_label_ids.append(label_ids)
        batch_label_ids = torch.stack(batch_label_ids, dim=0)
        batch_inputs['label_ids'] = batch_label_ids
        return batch_inputs
    
    
    def bio_transform(self, examples):
        pass   