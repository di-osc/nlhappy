from functools import lru_cache
from ..utils.make_datamodule import PLMBaseDataModule, char_idx_to_token
import pandas as pd
import numpy as np
import torch


class EntityExtractionDataModule(PLMBaseDataModule):
    """实体抽取数据模块
    可以解决嵌套实体和非连续实体
    """
    def __init__(self,
                 dataset: str,
                 batch_size: int,
                 transform: str = 'w2ner',
                 plm: str = 'roberta-wwm-base',
                 **kwargs):
        super().__init__()
        self.transforms['w2ner'] = self.w2ner_transform
        self.transforms['globalpointer'] = self.gp_transform


    @staticmethod
    def show_one_sample(self):
        return '{"text":"这是一个长颈鹿","entities":[{"indexes":[4,5,6],"label":"动物", "text":"长颈鹿"}]}'


    @property
    @lru_cache()
    def label2id(self):
        labels = sorted(pd.Series(np.concatenate(self.train_df.entities)).apply(lambda x: x['label']).drop_duplicates().values)
        return {l:i for i,l in enumerate(labels)}


    @property
    def id2label(self):
        return {i:l for l,i in self.label2id.items()}


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
        batch_ents = examples['entities']
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
        # batch_dist_ids = []
        dis2idx = self.get_dis2idx()
        for i, text in enumerate(batch_text):
            mapping = batch_mappings[i]
            ents = batch_ents[i]
            length = batch_lengths[i]
            label_ids = np.zeros((max_length, max_length), dtype=np.int)
            for ent in ents:
                idx_ls = ent['indexes']
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


    def gp_transform(self, examples):
        batch_text = examples['text']
        batch_ents = examples['entities']
        max_length = self.get_max_length()
        batch_inputs = {'input_ids':[],'token_type_ids':[],'attention_mask':[]}
        batch_label_ids = []
        for i, text in enumerate(batch_text):
            inputs = self.tokenizer(text, 
                                    padding='max_length',  
                                    max_length=max_length,
                                    truncation=True,
                                    return_offsets_mapping=True)
            ents = batch_ents[i]
            mapping = inputs['offset_mapping']
            label_ids = torch.zeros(len(self.hparams.label2id), max_length, max_length)
            for ent in ents :
                # +1 是因为添加了 [CLS]
                start = ent['indexes'][0]
                start = char_idx_to_token(start, mapping)
                end = ent['indexes'][-1] 
                end = char_idx_to_token(end, mapping)
                label_id = self.hparams.label2id[ent['label']]
                label_ids[label_id,  start, end] = 1
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_label_ids.append(label_ids)
        batch_label_ids = torch.stack(batch_label_ids, dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        batch['label_ids'] = batch_label_ids
        return batch


    def setup(self, stage):
        self.hparams.max_length = self.get_max_length()
        if self.hparams.transform == 'w2ner':
            labels = list(self.label2id.keys())
            label2id = {'<pad>':0, '<suc>':1}
            for i, label in enumerate(labels):
                label2id[label] = i+2
            id2label = {i:l for l,i in label2id.items()}
            self.hparams.label2id = label2id
            self.hparams.id2label = id2label
        else:
            self.hparams.label2id = self.label2id
            self.hparams.id2label = self.id2label
        self.dataset.set_transform(self.transforms.get(self.hparams.transform))
        