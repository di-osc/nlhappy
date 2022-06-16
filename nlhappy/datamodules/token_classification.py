import pytorch_lightning as pl
from ..utils.make_datamodule import OSSStorer
from datasets import load_from_disk
from transformers import BertTokenizer, BertConfig, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from ..utils.preprocessing import fine_grade_tokenize
import torch


class TokenClassificationDataModule(pl.LightningDataModule):
    '''序列标注数据模块
    数据集字段:
    - tokens: list[str], 例如: ['我', '是', '一', '中', '国', '人', '。']
    - labels: list[str], 例如: ['O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']
    '''
    def __init__(
        self,
        dataset: str,
        plm: str,
        max_length: int,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        pretrained_dir: str ,
        data_dir: str ,
        label_pad_id: int = -100
        ):
        super().__init__()

        self.save_hyperparameters()

        
        
    def prepare_data(self):
        '''下载数据集和预训练模型'''
        oss = OSSStorer()
        oss.download_dataset(self.hparams.dataset, self.hparams.data_dir)
        oss.download_plm(self.hparams.plm, self.hparams.pretrained_dir)

    def transform(self, examples):
        batch = {'inputs':[], 'label_ids':[]}
        for i, t_list in enumerate(examples['tokens']):
            tokens = [t['text'] for t in t_list]
            label_ids = [self.hparams['label2id'][t['label']] for t in t_list]
            inputs = self.tokenizer.encode_plus(
                tokens,  
                add_special_tokens=True,
                padding='max_length',  
                max_length=self.hparams.max_length,
                truncation=True)
            inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
            if len(label_ids) < (self.hparams.max_length-2):
                O_id = self.hparams.label2id['O']
                label_ids = [O_id] + label_ids + [O_id] + [-100]*(self.hparams.max_length-len(label_ids)-2)
            else: label_ids = [O_id] + label_ids + [O_id]
            batch['inputs'].append(inputs)
            label_ids = torch.tensor(label_ids)
            batch['label_ids'].append(label_ids)
        return batch



    def setup(self, stage):
        self.dataset = load_from_disk(self.hparams.data_dir + self.hparams.dataset)
        set_labels = sorted(set([t['label'] for t_list in self.dataset['train']['tokens'] for t in t_list]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams.label2id = label2id
        self.hparams.id2label = id2label
        self.dataset.set_transform(transform=self.transform)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_dir +self.hparams.plm)
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        trf_config = AutoConfig.from_pretrained(self.hparams.pretrained_dir + self.hparams.plm)
        self.hparams.trf_config = trf_config
        


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
        if self.test_dataset is not None:
            return DataLoader(dataset=self.dataset['test'],
                              batch_size=self.hparams.batch_size,
                              shuffle=False,
                              pin_memory=self.hparams.pin_memory,
                              num_workers=self.hparams.num_workers)




