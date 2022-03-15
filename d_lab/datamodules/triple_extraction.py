import pytorch_lightning as pl
from ..utils.storer import OSSStorer
from ..utils.preprocessing import fine_grade_tokenize
import torch
from datasets import load_from_disk
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from typing import Dict

class TripleExtractionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        plm: str,
        max_length: int ,
        batch_size: int ,
        num_workers: int ,
        pin_memory: bool,
        data_dir: str ,
        pretrained_dir: str 
        ):  
        super().__init__()

        self.save_hyperparameters()


    def prepare_data(self):
        '''下载数据集和预训练模型.'''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, localpath=self.hparams.data_dir)
        oss.download_plm(self.hparams.plm, localpath=self.hparams.pretrained_dir)

    
    def setup(self, stage: str) -> None:
        """读取数据集, 对数据设置转换"""
        # 加载数据
        self.dataset = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
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
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir + self.hparams.plm)
        self.bert_config = BertConfig.from_pretrained(self.hparams.pretrained_dir + self.hparams.plm)
        self.hparams['bert_config'] = self.bert_config
        self.hparams['token2id'] = dict(self.tokenizer.vocab)
        # 设置数据转换
        self.dataset.set_transform(transform=self.transform)


    def transform(self, example) -> Dict:
        batch_text = example['text']
        batch_triples = example['triples']
        batch_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'so_ids': [], 'head_ids': [], 'tail_ids': []}
        for i, text in enumerate(batch_text):
            tokens = fine_grade_tokenize(text, self.tokenizer)
            inputs = self.tokenizer.encode_plus(
                tokens, 
                padding='max_length',  
                max_length=self.hparams.max_length,
                add_special_tokens=True,
                truncation=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            triples = batch_triples[i]
            so_ids = torch.zeros(2, self.hparams.max_length, self.hparams.max_length)
            # span_ids = torch.zeros(len(self.hparams['s_label2id']), self.hparams.max_length, self.hparams.max_length)
            head_ids = torch.zeros(len(self.hparams['p_label2id']), self.hparams.max_length, self.hparams.max_length)
            tail_ids = torch.zeros(len(self.hparams['p_label2id']), self.hparams.max_length, self.hparams.max_length)
            for triple in triples:
                #加1是因为有cls
                so_ids[0][triple['subject']['offset'][0] + 1][triple['subject']['offset'][1]] = 1
                so_ids[1][triple['object']['offset'][0] + 1][triple['object']['offset'][1]] = 1
                # span_ids[self.hparams['s_label2id'][triple['object']['label']]][triple['object']['offset'][0]+1][triple['object']['offset'][1]] = 1
                # span_ids[self.hparams['s_label2id'][triple['subject']['label']]][triple['subject']['offset'][0]+1][triple['object']['offset'][1]] = 1
                head_ids[self.hparams['p_label2id'][triple['predicate']]][triple['subject']['offset'][0]+1][triple['object']['offset'][0]+1] = 1
                tail_ids[self.hparams['p_label2id'][triple['predicate']]][triple['subject']['offset'][1]][triple['object']['offset'][1]] = 1
            batch_inputs['so_ids'].append(so_ids)
            # batch_inputs['span_ids'].append(span_ids)
            batch_inputs['head_ids'].append(head_ids)
            batch_inputs['tail_ids'].append(tail_ids)
        batch_inputs['so_ids'] = torch.stack(batch_inputs['so_ids'], dim=0)
        # batch_inputs['span_ids'] = torch.stack(batch_inputs['span_ids'], dim=0)
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