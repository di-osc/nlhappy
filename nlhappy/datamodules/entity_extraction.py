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
                 plm: str = 'hfl/chinese-roberta-wwm-ext',
                 **kwargs):
        super().__init__()
        
    def setup(self, stage: str) -> None:
        self.hparams.id2ent = self.id2ent
        self.hparams.id2bio = self.id2bio
         
    
    @property
    @lru_cache()
    def ent_labels(self) -> List:
        labels = sorted(pd.Series(np.concatenate(self.train_df['ents'])).apply(lambda x: x['label']).drop_duplicates().values)
        return labels
    
    @property
    def ent2id(self):
        return {l:i for i,l in enumerate(self.ent_labels)}

    @property
    def id2ent(self):
        return {i:l for l,i in self.ent2id.items()}
    
    @property
    @lru_cache()
    def bio_labels(self) -> List:
        b_labels = ['B' + '-' + l for l in self.ent_labels]
        i_labels = ['I' + '-' + l for l in self.ent_labels]
        return ['O'] + b_labels + i_labels
    
    @property
    def bio2id(self):
        return {l:i for i,l in enumerate(self.bio_labels)}
    
    @property
    def id2bio(self):
        return {i:l for l,i in self.bio2id.items()}

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
        max_length = self.get_max_length()
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
                label_ids[char_idx_to_token(idx_ls[-1], mapping), char_idx_to_token(idx_ls[0], mapping)] = self.ent2id[ent['label']]
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
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_inputs = self.tokenizer(batch_text,
                                      max_length=max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_token_type_ids=False,
                                      return_tensors='pt')
        batch_tag_ids = []
        for i, text in enumerate(batch_text):
            ents = batch_ents[i]
            tag_ids = torch.zeros(len(self.ent_labels), max_length, max_length)
            for ent in ents :
                if len(ent['indices']) == 0:
                    log.warn(f'found empty entity indexes in {text}')
                    continue
                start_char = ent['indices'][0]
                start = batch_inputs.char_to_token(i, start_char)
                end_char = ent['indices'][-1] 
                end = batch_inputs.char_to_token(i, end_char)
                label_id = self.ent2id[ent['label']]
                tag_ids[label_id,  start, end] = 1
            batch_tag_ids.append(tag_ids)
        batch_tag_ids = torch.stack(batch_tag_ids, dim=0)
        batch_inputs['tag_ids'] = batch_tag_ids
        return batch_inputs
    
    
    def bio_transform(self, examples):
        batch_text = examples['text']  
        max_length = self.get_batch_max_length(batch_text=batch_text)
        inputs = self.tokenizer(batch_text,
                                padding='max_length',
                                max_length=max_length,
                                truncation=True,
                                return_token_type_ids=False,
                                return_tensors='pt')
        batch_mask = inputs['attention_mask']
        batch_tags = torch.zeros_like(batch_mask, dtype=torch.long)
        batch_tags = batch_tags + (batch_mask - 1) * 100 # 将pad的部分改为-100
        for i, text in enumerate(batch_text):
            ents = examples['ents'][i]
            for ent in ents:
                indices = ent['indices']
                start_char = indices[0]
                start_token = inputs.char_to_token(i, start_char)
                batch_tags[i][start_token] = self.bio2id['B' + '-' + ent['label']]
                if len(indices) > 1:
                    end_char = indices[-1]
                    end_token = inputs.char_to_token(i, end_char)
                    batch_tags[i][start_token: end_token+1] = self.bio2id['I' + '-' + ent['label']]
        inputs['tag_ids'] = batch_tags
        return inputs