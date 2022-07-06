import pytorch_lightning as pl
from ..utils.make_datamodule import align_char_span, prepare_data_from_oss
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from typing import Dict
import os
import logging

log = logging.getLogger(__name__)

class TripleExtractionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        plm: str,
        max_length: int ,
        batch_size: int ,
        num_workers: int ,
        pin_memory: bool,
        dataset_dir: str ,
        plm_dir: str):
        """三元组抽取数据模块,
        数据集样式:
            {'text':'产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。
            'triples': [{'object': {'offset': [14, 20], 'text': '轻度情绪失调'},'predicate': '鉴别诊断','subject': {'offset': [0, 5], 'text': '产后抑郁症'}}]}}

        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            max_length (int): 包括特殊token的最大序列长度
            batch_size (int): 训练,验证,测试数据集的批次大小,
            num_workers (int): 多进程数
            pin_memory (bool): 是否应用锁页内存,
            dataset_dir (str): 数据集默认路径
            plm_dir (str): 预训练模型默认路径
        """
        super().__init__()

        self.save_hyperparameters()


    def prepare_data(self):
        '''下载数据集和预训练模型.'''
        prepare_data_from_oss(dataset=self.hparams.dataset,
                              plm=self.hparams.plm,
                              dataset_dir=self.hparams.dataset_dir,
                              plm_dir=self.hparams.plm_dir)

    
    def setup(self, stage: str) -> None:
        """读取数据集, 对数据设置转换"""
        # 加载数据
        self.dataset = load_from_disk(self.hparams.dataset_dir + self.hparams.dataset)
        # 筛选predicate标签, 构建标签映射
        p_labels = sorted(set([triple['predicate'] for triples in self.dataset['train']['triples'] for triple in triples]))
        p_label2id = {label: i for i, label in enumerate(p_labels)}
        p_id2label = {i: label for label, i in p_label2id.items()}
        self.hparams['p_label2id'] = p_label2id
        self.hparams['p_id2label'] = p_id2label
        # 筛选实体类型
        # sub_labels = set([triple['subject']['label'] for triples in self.dataset['train']['triples'] for triple in triples])
        # obj_labels = set([triple['object']['label'] for triples in self.dataset['train']['triples'] for triple in triples])
        # s_labels = sorted(sub_labels | obj_labels)
        # s_label2id = {label: i for i, label in enumerate(s_labels)}
        # self.hparams['s_label2id'] = s_label2id
        # self.hparams['s_id2label'] = {i: label for label, i in s_label2id.items()}
        # 读取vocab 和 bert配置文件
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        trf_config = AutoConfig.from_pretrained(plm_path)
        self.hparams['trf_config'] = trf_config
        # 设置数据转换
        self.dataset.set_transform(transform=self.transform)


    def transform(self, example) -> Dict:
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
            head_ids = torch.zeros(len(self.hparams['p_label2id']), self.hparams.max_length, self.hparams.max_length)
            tail_ids = torch.zeros(len(self.hparams['p_label2id']), self.hparams.max_length, self.hparams.max_length)
            for triple in triples:
                #加1是因为有cls
                try:
                    sub_start = triple['subject']['offset'][0]
                    sub_end = triple['subject']['offset'][1]
                    sub_start, sub_end = align_char_span((sub_start, sub_end), offset_mapping)
                    obj_start = triple['object']['offset'][0]
                    obj_end = triple['object']['offset'][1]
                    obj_start, obj_end = align_char_span((obj_start,obj_end), offset_mapping)
                    so_ids[0][sub_start + 1][sub_end] = 1
                    so_ids[1][obj_start + 1][obj_end] = 1
                    head_ids[self.hparams['p_label2id'][triple['predicate']]][sub_start+1][obj_start+1] = 1
                    tail_ids[self.hparams['p_label2id'][triple['predicate']]][sub_end][obj_end] = 1
                except:
                    log.warning('char offset align to token offset failed')
                    pass
            batch_inputs['so_ids'].append(so_ids)
            batch_inputs['head_ids'].append(head_ids)
            batch_inputs['tail_ids'].append(tail_ids)
        batch_inputs['so_ids'] = torch.stack(batch_inputs['so_ids'], dim=0)
        batch_inputs['head_ids'] = torch.stack(batch_inputs['head_ids'], dim=0)
        batch_inputs['tail_ids'] = torch.stack(batch_inputs['tail_ids'], dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        return batch


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