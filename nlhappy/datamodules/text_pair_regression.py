import torch
import pytorch_lightning as pl
from typing import Optional,  Tuple, List
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
import os
from ..utils.storer import OSSStorer



class TextPairRegressionDataModule(pl.LightningDataModule):
    '''文本对相似度数据模块
    参数:
    - dataset: 数据集名称, 文件为datasets.DataSet  feature 必须包含 text_a, text_b, label
    - plm: 预训练模型名称
    - max_length: 单个句子的最大长度
    - batch_size: 批次大小
    - num_workers: 加载数据时的线程数
    - pin_memory: 是否将数据加载到GPU
    - dataset_dir: 数据集所在的目录
    - plm_dir: 预训练模型所在的目录
    '''
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int,
                batch_size: int,
                num_workers: int = 2,
                pin_memory: bool =True,
                dataset_dir = 'datasets/',
                plm_dir = 'plms/'):  
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)

    
    def prepare_data(self):
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.dataset_dir)
        oss.download_plm(self.hparams.plm, self.hparams.plm_dir)

        
    def setup(self, stage: str):
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        self.dataset = load_from_disk(dataset_path)
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        self.hparams['trf_config'] = AutoConfig.from_pretrained(plm_path)
        self.dataset.set_transform(transform=self.transform)

    def transform(self, examples):
        batch_text_a = examples['text_a']
        batch_text_b = examples['text_b']
        similarities = examples['similarity']
        batch = {'inputs_a': [], 'inputs_b': [], 'similarities':[]}
        for i  in range(len(batch_text_a)):
            inputs_a= self.tokenizer(batch_text_a[i], 
                                    padding='max_length', 
                                    max_length=self.hparams.max_length, 
                                    truncation=True)
            inputs_a = dict(zip(inputs_a.keys(), map(torch.tensor, inputs_a.values())))
            batch['inputs_a'].append(inputs_a)
            inputs_b = self.tokenizer(batch_text_b[i],
                                    padding='max_length', 
                                    max_length=self.hparams.max_length, 
                                    truncation=True)
            inputs_b = dict(zip(inputs_b.keys(), map(torch.tensor, inputs_b.values())))
            batch['inputs_b'].append(inputs_b)
            batch['similarities'].append(torch.tensor(similarities[i], dtype=torch.float))
        
        return batch
        


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
    @property
    def example(self):
        return """{'text_a': '左膝退变伴游离体','text_b': '单侧膝关节骨性关节病','similarity': 0}"""