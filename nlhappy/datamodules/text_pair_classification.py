import pytorch_lightning as pl
from typing import Optional,  Tuple, List, Union
from transformers import AutoConfig, AutoTokenizer
from ..utils.make_datamodule import prepare_data_from_oss
from torch.utils.data import DataLoader
import torch
import os
from datasets import load_from_disk, Dataset, DatasetDict 
from ..utils import utils



class TextPairClassificationDataModule(pl.LightningDataModule):
    '''句子对数据模块,用来构建pytorch_lightning的数据模块.
    dataset_example:
        {'text_a': '肺结核','text_b': '关节炎','label': 不是一种病}
    '''
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int,
                batch_size: int,
                return_pair: bool=False,
                num_workers: int = 0,
                pin_memory: bool =False,
                dataset_dir = './datasets/',
                plm_dir = './plms/'): 
        """参数:
        - dataset: 数据集名称,feature 必须包含 text_a, text_b, label
        - plm: 预训练模型名称
        - max_length: 单个句子的最大长度
        - batch_size: 批大小
        - return_pair: 是否以文本对为输入
        - num_workers: 加载数据时的线程数
        - pin_memory: 是否将数据加载到GPU
        - dataset_dir: 数据集所在的目录
        - plm_dir: 预训练模型所在的目录
        """ 
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)
        
        

    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        prepare_data_from_oss(dataset=self.hparams.dataset,
                              plm=self.hparams.plm,
                              dataset_dir=self.hparams.dataset_dir,
                              plm_dir=self.hparams.plm_dir)
        
    
    def setup(self, stage: str):
        dataset_path = os.path.join(self.hparams.dataset_dir , self.hparams.dataset)
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        self.dataset = load_from_disk(dataset_path=dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        set_labels = sorted(set([label for label in self.dataset['train']['label']]))
        self.hparams['label2id'] = {label: i for i, label in enumerate(set_labels)}
        self.hparams['id2label'] = {i: label for i, label in enumerate(set_labels)}
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        self.hparams['trf_config'] = AutoConfig.from_pretrained(plm_path)
        self.dataset.set_transform(transform=self.transform)
        
    def transform(self, examples):
        batch_text_a = examples['text_a']
        batch_text_b = examples['text_b']
        batch_labels = examples['label']
        batch = {'inputs_a': [], 'inputs_b': [], 'label_ids':[]}
        batch_cross = {'inputs': [], 'label_ids':[]}
        if self.hparams.return_pair:
            for i  in range(len(batch_text_a)):
                inputs_a= self.tokenizer(batch_text_a[i], 
                                        padding='max_length', 
                                        max_length=self.hparams.max_length+2, 
                                        truncation=True)
                inputs_a = dict(zip(inputs_a.keys(), map(torch.tensor, inputs_a.values())))
                batch['inputs_a'].append(inputs_a)
                inputs_b = self.tokenizer(batch_text_b[i],
                                        padding='max_length', 
                                        max_length=self.hparams.max_length, 
                                        truncation=True)
                inputs_b = dict(zip(inputs_b.keys(), map(torch.tensor, inputs_b.values())))
                batch['inputs_b'].append(inputs_b)
                batch['label_ids'].append(self.hparams['label2id'][batch_labels[i]])
            return batch
        else:
            for i in range(len(batch_text_a)):
                inputs = self.tokenizer(batch_text_a[i],
                                        batch_text_b[i], 
                                        padding='max_length', 
                                        max_length=self.hparams.max_length*2+2, 
                                        truncation=True)
                inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
                batch_cross['inputs'].append(inputs)
                batch_cross['label_ids'].append(self.hparams['label2id'][batch_labels[i]])
            return batch_cross


    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(dataset= self.dataset['train'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(dataset=self.dataset['validation'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(dataset=self.dataset['test'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False)
        


        