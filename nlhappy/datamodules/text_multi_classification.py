import imp
import pytorch_lightning as pl
from typing import Any
from ..utils.make_datamodule import OSSStorer
from torch.utils.data import DataLoader
from datasets import load_from_disk
import os

class TextMultiClassification(pl.LightningModule):
    """多标签多分类的数据模块, 每个标签下有多重选择的情况"""
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int,
                batch_size: int,
                pin_memory: bool=False,
                num_workers: int=0,
                data_dir: str ='./datasets',
                plm_dir: str = './plms'):
        """多标签多分类数据模块, 每个标签下游多重选择

        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            max_length (int): 单文本最大长度
            batch_size (int): 批次大小
            pin_memory (bool, optional): _description_. Defaults to False.
            num_workers (int, optional): _description_. Defaults to 0.
            data_dir (str, optional): _description_. Defaults to './datasets'.
            plm_dir (str, optional): _description_. Defaults to './plms'.
        """
        super().__init__()
        self.save_hyperparameters()


    def prepare_data(self) -> None:
        '下载数据集和预训练模型'
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.data_dir)
        oss.download_plm(self.hparams.plm, self.hparams.plm_dir)

    def transform(self, batch):
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        """需要设置参数label2id, id2label, token2id, bert_config最后要对dataset设置transform"""
        raise NotImplementedError
    
    @property
    def dataset(self):
        return load_from_disk(self.hparams.data_dir + self.hparams.dataset)
    


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
