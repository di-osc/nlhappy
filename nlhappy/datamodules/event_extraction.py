from ..utils.make_datamodule import PLMBaseDataModule, get_logger, char_idx_to_token
from functools import lru_cache
import torch
import pandas as pd
import numpy as np


log = get_logger()

class EventExtractionDataModule(PLMBaseDataModule):
    """事件抽取数据模块

    数据集格式:
        {'text':'油服巨头哈里伯顿裁员650人 因美国油气开采活动放缓',
        'events':[{'label':'裁员事件','trigger':{'offset':(8,10), 'text':裁员'}, 'roles':[{'label':'裁员方', 'offset':(4,8), 'text':'哈里伯顿'}, {'label':'裁员人数', 'offset':(10, 14), 'text':'650人'}]}]}
    """
    def __init__(self, 
                dataset: str, 
                plm: str, 
                batch_size: int,
                auto_length: int = 'max',
                transform: str = 'gplinker',
                **kwargs):
        super().__init__()
        self.transforms = {'gplinker': self.gplinker_transform,
                           'blinker': self.blinker_transform}


    def setup(self, stage: str = 'fit') -> None:
        self.hparams.max_length = self.get_max_length()
        if self.hparams.transform == 'gplinker':
            self.hparams.id2label = {i:l for l,i in self.combined2id.items()}
            self.dataset.set_transform(self.gplinker_transform)
        elif self.hparams.transform == 'blinker':
            self.hparams.id2label = {i:l for l,i in self.ent2id.items()}
            self.dataset.set_transform(self.blinker_transform)
            
            
    @property
    @lru_cache()
    def event_labels(self):
        labels = pd.Series(np.concatenate(self.train_df.events.values)).apply(lambda x: x['label']).drop_duplicates().values
        return labels.tolist()
    
    
    @property
    def event2id(self):
        return {l:i for i, l in enumerate(self.event_labels)}
    
    @property
    def trigger_labels(self):
        "将触发词改为 XX事件-触发词 这种样式"
        return [e + '-' + '触发词' for e in self.event_labels]
    
    
    @property
    @lru_cache()
    def role_labels(self):
        labels = pd.Series(np.concatenate(pd.Series(np.concatenate(self.train_df.events.values)).apply(lambda x: x['roles']).values)).apply(lambda x: x['label']).drop_duplicates().values
        return labels.tolist()
    
    
    @property
    def role2id(self):
        return {l:i for i, l in enumerate(self.role_labels)}
    
    
    @property
    def ent_labels(self):
        """将触发词与各个事件角色当做实体"""
        return sorted(self.trigger_labels + self.role_labels)
    
    @property
    def ent2id(self):
        return {l:i for i,l in enumerate(self.ent_labels)}


    @property
    @lru_cache()
    def combined_labels(self):
        """将事件类型跟所有事件角色(包括触发词)拼接为一个标签
        - 例如: 裁员事件-触发词, 裁员事件-裁员方
        """
        def get_labels(e):
            labels = []
            label = e['label']
            labels.append((label, '触发词')) #触发词同样添加进去
            for r in e['roles']:
                pair_label = (label,r['label'])
                if pair_label not in labels:
                    labels.append(pair_label)
            return labels
        df = pd.DataFrame(np.concatenate(pd.Series(np.concatenate(self.train_df.events.values)).apply(get_labels).values).tolist()).drop_duplicates()
        labels = sorted([tuple(v) for v in df.values])
        return labels


    @property
    def combined2id(self):
        return {l:i for i,l in enumerate(self.combined_labels)}
    
    @property
    def id2combined(self):
        return {i:l for i,l in enumerate(self.combined_labels)}


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
                                      return_tensors='pt',
                                      return_token_type_ids=False)
        batch_mappings = batch_inputs.pop('offset_mapping').tolist()
        for i, text in enumerate(batch_text):
            offset_mapping = batch_mappings[i]
            events = batch_events[i]
            role_ids = torch.zeros(len(self.combined2id), max_length, max_length, dtype=torch.long)
            head_ids = torch.zeros(1, max_length, max_length, dtype=torch.long)
            tail_ids = torch.zeros(1, max_length, max_length, dtype=torch.long)
            for event in events:
                e_label = event['label']
                roles = event['roles']
                # 将触发词当做事件角色之一
                trigger = event['trigger']
                trigger_head = trigger['offset'][0]
                trigger_tail = trigger['offset'][1]
                roles.append({'label':'触发词', 'offset': (trigger_head, trigger_tail)})
                for i, role1 in enumerate(roles):
                    label = (e_label, role1['label'])
                    role1_head = role1['offset'][0]
                    role1_tail = role1['offset'][1]-1
                    try:
                        _role1_head = char_idx_to_token(role1_head, offset_mapping=offset_mapping)
                        _role1_tail = char_idx_to_token(role1_tail, offset_mapping=offset_mapping)
                        role_ids[self.combined2id[label]][_role1_head][_role1_tail] = 1
                    except:
                        log.warning(f'role {role1["text"]} offset {(role1_head, role1_tail)} align to token offset failed in \n\t {text}')
                        continue
                        # 这个role 跟 其他的每个role 头头 尾尾 联系起来
                    for j, role2 in enumerate(roles):
                        if j>i:
                            role2_head = role2['offset'][0]
                            role2_tail = role2['offset'][1]-1
                            try:
                                _role2_head = char_idx_to_token(role2_head, offset_mapping=offset_mapping)
                                _role2_tail = char_idx_to_token(role2_tail, offset_mapping=offset_mapping)
                                head_ids[0][min(_role1_head, _role2_head)][max(_role1_head, _role2_head)] = 1
                                tail_ids[0][min(_role1_tail, _role2_tail)][max(_role1_tail, _role2_tail)] = 1
                            except:
                                log.warning(f'role {role2["text"]} offset {(role2_head, role2_tail)} align to token offset failed in \n\t {text}')
                                continue
            batch_role_ids.append(role_ids)
            batch_head_ids.append(head_ids)
            batch_tail_ids.append(tail_ids)
        batch_inputs['role_ids'] = torch.stack(batch_role_ids, dim=0)
        batch_inputs['head_ids'] = torch.stack(batch_head_ids, dim=0)
        batch_inputs['tail_ids'] = torch.stack(batch_tail_ids, dim=0)
        return batch_inputs
    
    def blinker_transform(self, examples):
        batch_text = examples['text']
        batch_events = examples['events']
        batch_ent_tags = []
        batch_head_tags = []
        batch_tail_tags = []
        max_length = self.get_max_length()
        batch_inputs = self.tokenizer(batch_text,
                                      max_length=max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_offsets_mapping=True,
                                      return_tensors='pt',
                                      return_token_type_ids=False)
        batch_mappings = batch_inputs.pop('offset_mapping').tolist()
        for i, text in enumerate(batch_text):
            offset_mapping = batch_mappings[i]
            events = batch_events[i]
            ent_tags = torch.zeros(len(self.ent2id), max_length, max_length, dtype=torch.long)
            head_tags = torch.zeros(1, max_length, max_length, dtype=torch.long)
            tail_tags = torch.zeros(1, max_length, max_length, dtype=torch.long)
            for event in events:
                e_label = event['label']
                try:
                    trigger = event['trigger']
                    trigger_label = e_label + '-' + '触发词'
                    trigger_head = trigger['offset'][0]
                    trigger_tail = trigger['offset'][1]-1
                    _trigger_head = char_idx_to_token(trigger_head, offset_mapping=offset_mapping)
                    _trigger_tail = char_idx_to_token(trigger_tail, offset_mapping=offset_mapping)
                    ent_tags[self.ent2id[trigger_label]][_trigger_head][_trigger_tail] = 1
                except:
                    log.warning(f'set triiger({trigger}) to tag failed in \n\t {text}')
                    pass
                for i, role1 in enumerate(event['roles']):
                    label = role1['label']
                    role1_head = role1['offset'][0]
                    role1_tail = role1['offset'][1]-1
                    try:
                        _role1_head = char_idx_to_token(role1_head, offset_mapping=offset_mapping)
                        _role1_tail = char_idx_to_token(role1_tail, offset_mapping=offset_mapping)
                        ent_tags[self.ent2id[label]][_role1_head][_role1_tail] = 1
                    except:
                        log.warning(f'role {role1["text"]} offset {(role1_head, role1_tail)} align to token offset failed in \n\t {text}')
                        continue
                        # 这个role 跟 其他的每个role 头头 尾尾 联系起来
                    for j, role2 in enumerate(event['roles']):
                        if j>i:
                            role2_head = role2['offset'][0]
                            role2_tail = role2['offset'][1]-1
                            try:
                                _role2_head = char_idx_to_token(role2_head, offset_mapping=offset_mapping)
                                _role2_tail = char_idx_to_token(role2_tail, offset_mapping=offset_mapping)
                                head_tags[0][min(_role1_head, _role2_head)][max(_role1_head, _role2_head)] = 1
                                tail_tags[0][min(_role1_tail, _role2_tail)][max(_role1_tail, _role2_tail)] = 1
                            except:
                                log.warning(f'role {role2["text"]} offset {(role2_head, role2_tail)} align to token offset failed in \n\t {text}')
                                continue
            batch_ent_tags.append(ent_tags)
            batch_head_tags.append(head_tags)
            batch_tail_tags.append(tail_tags)
        batch_inputs['ent_tags'] = torch.stack(batch_ent_tags, dim=0)
        batch_inputs['head_tags'] = torch.stack(batch_head_tags, dim=0)
        batch_inputs['tail_tags'] = torch.stack(batch_tail_tags, dim=0)
        return batch_inputs