import pytorch_lightning as pl
from ..utils.storer import OSSStorer
import os
from datasets import load_from_disk, DatasetDict
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from ..tokenizers import load_vocab
from ..utils import utils

log = utils.get_logger(__name__)


class PLMDataModule(pl.LightningDataModule):
    """预训练语言模型的数据模块基类,需要重写transform和label2id两个方法即可
    """
    def __init__(
        self,
        dataset: str,
        plm: str,
        batch_size: int,
        pin_memory: bool=False,
        num_workers: int=0,
        data_dir: str ='./datasets/',
        plm_dir: str = './plms/',
        **kwargs
        ) :
        super().__init__()
        self.save_hyperparameters()


    def prepare_data(self) -> None:
        '下载数据集和预训练模型'
        oss = OSSStorer()
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        if os.path.exists(dataset_path):
            log.info(f'{dataset_path} already exists.')
        else:
            log.info('not exists dataset in {}'.format(dataset_path))
            log.info('start downloading dataset from oss')
            oss.download_dataset(self.hparams.dataset, self.hparams.dataset_dir)
        if os.path.exists(plm_path):
            log.info(f'{plm_path} already exists.') 
        else : 
            log.info('not exists plm in {}'.format(plm_path))
            log.info('start downloading plm from oss')
            oss.download_plm(self.hparams.plm, self.hparams.plm_dir)


    def setup(self, stage: str) -> None:
        """需要设置参数label2id, id2label, token2id, bert_config最后要对dataset设置transform"""
        self.dataset = load_from_disk(os.path.join(self.hparams.data_dir, self.hparams.dataset))
        self.hparams.token_vocab = self.token_vocab
        self.hparams.label_vocab = self.label_vocab
        self.hparams.trf_config = self.trf_config
        self.dataset.set_transform(self.transform)


    def transform(self, batch):
        raise NotImplementedError



    @property
    def token_vocab(self):
        return load_vocab(os.path.join(self.hparams.plm_dir, self.hparams.plm, 'vocab.txt'))


    @property
    def label_vocab(self):
        raise NotImplementedError


    @property
    def trf_config(self):
        return AutoConfig.from_pretrained(self.hparams.plm_dir+self.hparams.plm)


    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(os.path.join(self.hparams.plm_dir, self.hparams.plm))

    
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