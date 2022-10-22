from ..utils.make_datamodule import PLMBaseDataModule, get_logger, char_idx_to_token
from functools import lru_cache
import torch
import pandas as pd
import numpy as np


log = get_logger()

class EventExtractionDataModule(PLMBaseDataModule):
    """事件抽取数据模块

    数据集格式:
        {'text':'半年前，患者开始出现持续性头疼',
        'events':[{'label':'临床事件', 'roles':[{'label':'主体词', 'offset':(14,16), 'text':'头疼'},{'label':'描述词', 'offset':(10,13), 'text':'持续性'}]}]}
    """
    def __init__(self, 
                dataset: str, 
                plm: str, 
                batch_size: int,
                auto_length: int = 'max',
                transform: str = 'gplinker',
                **kwargs):
        super().__init__()
        self.transforms = {'gplinker': self.gplinker_transform}


    def setup(self, stage: str = 'fit') -> None:
        self.hparams.max_length = self.get_max_length()
        if self.hparams.transform == 'gplinker':
            self.hparams.id2label = {i:l for l,i in self.combined2id.items()}
            self.dataset.set_transform(self.gplinker_transform)


    @property
    @lru_cache()
    def event2id(self):
        labels = pd.Series(np.concatenate(self.train_df.events.values)).apply(lambda x: x['label']).drop_duplicates().values
        return {l:i for i, l in enumerate(labels)}

    @property
    @lru_cache()
    def role2id(self):
        labels = pd.Series(np.concatenate(pd.Series(np.concatenate(self.train_df.events.values)).apply(lambda x: x['roles']).values)).apply(lambda x: x['label']).drop_duplicates().values
        return {l:i for i, l in enumerate(labels)}

    @property
    @lru_cache()
    def combined2id(self):
        def get_labels(e):
            labels = []
            label = e['label']
            for r in e['roles']:
                labels.append((label,r['label']))
            return labels
        df = pd.DataFrame(np.concatenate(pd.Series(np.concatenate(self.train_df.events.values)).apply(get_labels).values).tolist()).drop_duplicates()
        labels = sorted([tuple(v) for v in df.values])
        return {l:i for i,l in enumerate(labels)}



    def gplinker_transform(self, examples):
        batch_text = examples['text']
        batch_events = examples['events']
        batch_role_ids = []
        batch_head_ids = []
        batch_tail_ids = []
        max_length = self.get_max_length()
        batch_inputs = self.tokenizer(batch_text,
                                      max_length=max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_offsets_mapping=True,
                                      return_tensors='pt')
        batch_mappings = batch_inputs.pop('offset_mapping').tolist()
        for i, text in enumerate(batch_text):
            offset_mapping = batch_mappings[i]
            events = batch_events[i]
            role_ids = torch.zeros(len(self.combined2id), max_length, max_length)
            head_ids = torch.zeros(1, max_length, max_length)
            tail_ids = torch.zeros(1, max_length, max_length)
            for event in events:
                e_label = event['label']
                for i, role in enumerate(event['roles']):
                    label = (e_label, role['label'])
                    try:
                        start_ = role['offset'][0]
                        end_ = role['offset'][1]-1
                        start = char_idx_to_token(start_, offset_mapping=offset_mapping)
                        end = char_idx_to_token(end_, offset_mapping=offset_mapping)
                    except:
                        log.warning(f'role {role["text"]} offset {(start, end)} align to token offset failed in \n{text}')
                        continue
                    role_ids[self.combined2id[label]][start][end] = 1
                        # 这个role 跟 其他的每个role 头头 尾尾 联系起来
                    for j, role1 in enumerate(event['roles']):
                        if j>i:
                            start1 = role1['offset'][0]
                            end1 = role1['offset'][1]-1
                            try:
                                start1 = char_idx_to_token(start1, offset_mapping=offset_mapping)
                                end1 = char_idx_to_token(end1, offset_mapping=offset_mapping)
                            except:
                                log.warning(f'role {role1["text"]} offset {(start1, end1)} align to token offset failed in \n{text}')
                                continue
                            head_ids[0][min(start, start1)][max(start, start1)] = 1
                            tail_ids[0][min(end, end1)][max(end, end1)] = 1
            batch_role_ids.append(role_ids)
            batch_head_ids.append(head_ids)
            batch_tail_ids.append(tail_ids)
        batch_inputs['role_ids'] = torch.stack(batch_role_ids, dim=0)
        batch_inputs['head_ids'] = torch.stack(batch_head_ids, dim=0)
        batch_inputs['tail_ids'] = torch.stack(batch_tail_ids, dim=0)
        return batch_inputs


    @staticmethod
    def get_one_example():
        return {'text':'半年前，患者开始出现持续性头疼',
                'events':[{'label':'临床事件', 'roles':[{'label':'主体词', 'offset':(14,16), 'text':'头疼'},
                                                       {'label':'描述词', 'offset':(10,13), 'text':'持续性'}]}]}