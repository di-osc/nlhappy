import pytorch_lightning as pl
from ..utils.make_datamodule import prepare_data_from_oss
import torch
from datasets import load_from_disk
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from ..utils.preprocessing import fine_grade_tokenize
import datasets


class SpanClassificationDataModule(pl.LightningDataModule):
    """span分类的数据模块 数据集必须有text, spans两个字段
    """
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int,
                batch_size: int,
                pin_memory: bool,
                num_workers: int,
                dataset_dir: str ='./datasets/',
                plm_dir: str = './plms/') :
        super().__init__()
        self.save_hyperparameters()


    def prepare_data(self) -> None:
        '下载数据集和预训练模型'
        prepare_data_from_oss(dataset=self.hparams.dataset,
                              plm=self.hparams.plm,
                              dataset_dir=self.hparams.dataset_dir,
                              plm_dir=self.hparams.plm_dir)

    def transform(self, example):
        batch_text = example['text']
        batch_spans = example['spans']
        max_length = self.hparams.max_length
        batch_inputs = {'input_ids':[],'token_type_ids':[],'attention_mask':[], 'span_ids':[]}
        for i, text in enumerate(batch_text):
            tokens = fine_grade_tokenize(text, self.tokenizer)
            inputs = self.tokenizer.encode_plus(
                tokens, 
                padding='max_length',  
                max_length=max_length,
                truncation=True)
            spans = batch_spans[i]
            span_ids = torch.zeros(len(self.hparams.label2id), max_length, max_length)
            for span in spans :
                # +1 是因为添加了 [CLS]
                start = span['offset'][0] + 1
                end = span['offset'][1] 
                label_id = self.hparams.label2id[span['label']]
                span_ids[label_id,  start, end] = 1
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['span_ids'].append(span_ids)
        batch_inputs['span_ids'] = torch.stack(batch_inputs['span_ids'], dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        return batch

    def setup(self, stage: str) -> None:
        if isinstance(self.hparams.dataset, datasets.DatasetDict):
            self.dataset = self.hparams.dataset
        elif isinstance(self.hparams.dataset, str):
            self.dataset = load_from_disk(self.hparams.dataset_dir + self.hparams.dataset)
        # self.dataset = load_from_disk(self.hparams.dataset_dir + self.hparams.dataset)
        set_labels = sorted(set([span['label'] for spans in self.dataset['train']['spans'] for span in spans]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self.hparams['label2id'] = label2id
        self.hparams['id2label'] = id2label
        self.dataset.set_transform(transform=self.transform)
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.plm_dir + self.hparams.plm)
        self.hparams['token2id'] = dict(self.tokenizer.vocab)
        bert_config = BertConfig.from_pretrained(self.hparams.plm_dir + self.hparams.plm)
        self.hparams['bert_config'] = bert_config
    


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