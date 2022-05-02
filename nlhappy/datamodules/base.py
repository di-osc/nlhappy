import pytorch_lightning as pl
from ..utils.storer import OSSStorer
import os
from datasets import load_from_disk, DatasetDict
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from ..tokenizers import load_vocab


class PLMDataModuleBase(pl.LightningDataModule):
    """预训练语言模型的数据模块基类,需要重写transform和label2id两个方法即可
    """
    def __init__(
        self,
        dataset: str,
        plm: str,
        batch_size: int,
        pin_memory: bool=True,
        num_workers: int=0,
        data_dir: str ='./datasets/',
        pretrained_dir: str = './plms/',
        **kwargs
        ) :
        super().__init__()
        self.save_hyperparameters()


    def prepare_data(self) -> None:
        '下载数据集和预训练模型'
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.data_dir)
        oss.download_plm(self.hparams.plm, self.hparams.pretrained_dir)


    def setup(self, stage: str) -> None:
        """需要设置参数label2id, id2label, token2id, bert_config最后要对dataset设置transform"""
        self.dataset = load_from_disk(os.path.join(self.hparams.data_dir, self.hparams.dataset))
        self.hparams.token2id = self.token2id
        self.hparams.label2id = self.label2id
        self.hparams.trf_config = self.trf_config
        self.dataset.set_transform(self.transform)


    def transform(self, batch):
        raise NotImplementedError



    @property
    def token2id(self):
        return load_vocab(os.path.join(self.hparams.pretrained_dir, self.hparams.plm, 'vocab.txt'))


    @property
    def label2id(self):
        raise NotImplementedError


    @property
    def trf_config(self):
        return AutoConfig.from_pretrained(self.hparams.pretrained_dir+self.hparams.plm)


    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(os.path.join(self.hparams.pretrained_dir, self.hparams.plm))

    
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