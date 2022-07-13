import pytorch_lightning as pl
from ..utils.make_datamodule import prepare_data_from_oss, char_idx_to_token
import torch
from datasets import load_from_disk
import os
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import logging

log = logging.getLogger()


class PromptSpanExtractionDataModule(pl.LightningDataModule):
    """
    Data module for the prompt span extraction task.

    Dataset examples:
    {'text':'北京是中国的首都', 'prompt':'国家', 'spans':[{'text': '中国', 'offset':(3, 5)}]}
    {'text':'北京是中国的首都', 'prompt':'中国的首都', 'spans':[{'text': '北京', 'offset':(0, 2)}]}
    {'text': '北京是中国的首都', 'prompt': '北京的国家', 'spans': [{'text': '中国', 'offset': (3, 5)}]}
    其中offset 为左闭右开的字符级别下标
    """
    
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int,
                batch_size: int,
                pin_memory: bool=False,
                num_workers: int=0,
                dataset_dir: str ='./datasets/',
                plm_dir: str = './plms/') :
        """基于模板提示的文本片段抽取数据模块

        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            max_length (int): 文本最大长度
            batch_size (int): 批次大小
            pin_memory (bool, optional): 锁页内存. Defaults to True.
            num_workers (int, optional): 多进程. Defaults to 0.
            dataset_dir (str, optional): 数据集目录. Defaults to './datasets/'.
            plm_dir (str, optional): 预训练模型目录. Defaults to './plms/'.
        """
        super().__init__()
        self.save_hyperparameters()
        
        
    def prepare_data(self) -> None:
        
        prepare_data_from_oss(dataset=self.hparams.dataset,
                              plm=self.hparams.plm,
                              dataset_dir=self.hparams.dataset_dir,
                              plm_dir=self.hparams.plm_dir)
        
        
    def transform(self, example):
        batch_text = example['text']
        batch_spans = example['spans']
        batch_prompt = example['prompt']
        max_length = self.hparams.max_length
        batch = {'inputs': [], 'span_ids': []}
        for i, text in enumerate(batch_text):
            # tokens = fine_grade_tokenize(text, self.tokenizer)
            prompt = batch_prompt[i]
            inputs = self.tokenizer(
                prompt, 
                text, 
                padding='max_length',  
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True)
            offset_mapping = [list(x) for x in inputs["offset_mapping"]]
            bias = 0
            for index in range(len(offset_mapping)):
                if index == 0:
                    continue
                mapping = offset_mapping[index]
                if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                    bias = index
                if mapping[0] == 0 and mapping[1] == 0:
                    continue
                offset_mapping[index][0] += bias
                offset_mapping[index][1] += bias
            spans = batch_spans[i]
            span_ids = torch.zeros(1, max_length, max_length)
            for span in spans:
                start = char_idx_to_token(span['offset'][0]+bias, offset_mapping)
                end = char_idx_to_token(span['offset'][1]-1+bias, offset_mapping)
                # 如果超出边界，则跳过
                try:
                    span_ids[0, start, end] = 1.0
                except Exception:
                    log.warning(f'set span {(start, end)} out of boudry')
                    pass 
            del inputs['offset_mapping']
            inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
            batch['inputs'].append(inputs)
            batch['span_ids'].append(span_ids)
        return batch
    
    
    def setup(self, stage: str) -> None:
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        self.dataset = load_from_disk(dataset_path)
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        trf_config = AutoConfig.from_pretrained(plm_path)
        self.hparams['trf_config'] = trf_config
        self.dataset.set_transform(transform=self.transform)
            

    def train_dataloader(self):
        '''
        返回训练集的DataLoader.
        '''
        return DataLoader(
            dataset= self.dataset['train'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=True)
        
    def val_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.dataset['validation'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)

    def test_dataloader(self):
        '''
        返回验证集的DataLoader.
        '''
        return DataLoader(
            dataset=self.dataset['test'], 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False)
    
    
    