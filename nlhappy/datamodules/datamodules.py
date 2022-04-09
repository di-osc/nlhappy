import pytorch_lightning as pl
from ..utils.storer import OSSStorer
import os
import torch
from datasets import load_from_disk
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader


class BertDataModule(pl.LightningDataModule):
    """以BERT为文本表示的数据模块的基类,一般只需要重写setup和transform两个方法即可
    """
    def __init__(
        self,
        dataset: str,
        plm: str,
        max_length: int,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        data_dir: str ='./datasets',
        pretrained_dir: str = './plms',
        ) :
        super().__init__()
        self.save_hyperparameters()


    def prepare_data(self) -> None:
        '下载数据集和预训练模型'
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.data_dir)
        oss.download_plm(self.hparams.plm, self.hparams.pretrained_dir)

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