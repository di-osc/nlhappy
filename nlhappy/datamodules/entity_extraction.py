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


    @staticmethod
    def show_one_sample(self):
        return '{"text":"这是一个长颈鹿","entities":[{"indexes":[4,5,6],"label":"动物"}]}'


    @property
    @lru_cache()
    def label2id(self):
        label2id = {'<pad>':0, '<suc>':1}
        labels = sorted(pd.Series(np.concatenate(self.train_df.entities)).apply(lambda x: x['label']).drop_duplicates().values)
        for i, l in enumerate(labels):
            label2id[l] = i+2
        return label2id


    @property
    def id2label(self):
        return {i:l for l,i in self.label2id.items()}


    @property
    @lru_cache()
    def dis2idx(self):
        dis2idx = torch.zeros((1000), dtype=torch.long)
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


    def w2ner_transform(self, examples):
        max_length = self.hparams.max_length
        batch_text = examples['text']
        batch_ents = examples['entities']
        batch_inputs = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[]}
        batch_label_ids = []
        batch_dist_ids = []
        for i, text in enumerate(batch_text):
            inputs = self.tokenizer(text,
                                    max_length=max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_offsets_mapping=True,
                                    add_special_tokens=False)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            mapping = inputs['offset_mapping']
            ents = batch_ents[i]
            label_ids = torch.zeros(max_length, max_length, dtype=torch.long)
            for ent in ents:
                idx_ls = ent['indexes']
                for i in range(len(idx_ls)):
                    if i +1 >= len(idx_ls):
                        break
                    label_ids[char_idx_to_token(idx_ls[i], mapping), char_idx_to_token(idx_ls[i+1], mapping)] = 1
                label_ids[char_idx_to_token(idx_ls[-1], mapping), char_idx_to_token(idx_ls[0], mapping)] = self.label2id[ent['label']]
            batch_label_ids.append(label_ids)
            dist_ids = torch.zeros(max_length, max_length, dtype=torch.long)
            for k in range(max_length):   #字的个数
                dist_ids[k, :] += k
                dist_ids[:, k] -= k     #dist_ids即位置输入，第一行只有一个0其余为负，第二行1,0，其余为负，第三行2，1,0，其余为负

            for i in range(max_length):
                for j in range(max_length):
                    if dist_ids[i, j] < 0:
                        dist_ids[i, j] = self.dis2idx[-dist_ids[i, j]] + 9
                    else:
                        dist_ids[i, j] = self.dis2idx[dist_ids[i, j]]
            batch_dist_ids.append(dist_ids)

        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        batch['label_ids'] = torch.stack(batch_label_ids)
        batch['distance_ids'] = torch.stack(batch_dist_ids)
        return batch


    def setup(self, stage):
        self.hparams.max_length = self.get_max_length()
        self.hparams.label2id = self.label2id
        self.hparams.id2label = self.id2label
        self.dataset.set_transform(self.transforms.get(self.hparams.transform))
        