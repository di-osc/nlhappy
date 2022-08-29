from ..utils.make_datamodule import align_char_span_text_b, PLMBaseDataModule
import torch
from typing import Dict, Union
import logging

log = logging.getLogger()



class PromptRelationExtractionDataModule(PLMBaseDataModule):
    def __init__(self,
                dataset: str,
                plm: str,
                auto_length: Union[str, int],
                transform: str,
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
        self.transforms = {'prompt_gplinker': self.gplinker_transform}        
        assert self.hparams.transform in self.transforms.keys(), f'availabel models for relation extraction: {self.transforms.keys()}'
    
        
    def setup(self, stage: str) -> None:
        self.hparams.max_length = self.get_max_length()
        self.dataset.set_transform(transform=self.transforms.get(self.hparams.transform))
        
        
    def gplinker_transform(self, example) -> Dict:
        batch_text = example['text']
        batch_triples = example['triples']
        batch_prompts = example['prompts']
        batch_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'so_ids': [], 'head_ids': [], 'tail_ids': []}
        for i, text in enumerate(batch_text):
            prompt = batch_prompts[i]
            inputs = self.tokenizer(
                prompt,
                text, 
                padding='max_length',  
                max_length=self.hparams.max_length,
                truncation=True,
                return_offsets_mapping=True)
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            offset_mapping = inputs['offset_mapping']
            so_ids = torch.zeros(2, self.hparams.max_length, self.hparams.max_length)
            head_ids = torch.zeros(1, self.hparams.max_length, self.hparams.max_length)
            tail_ids = torch.zeros(1, self.hparams.max_length, self.hparams.max_length)
            triples = batch_triples[i]
            for triple in triples:
                #加1是因为有cls
                sub_start = triple['subject']['offset'][0] 
                sub_end = triple['subject']['offset'][1]
                obj_start = triple['object']['offset'][0] 
                obj_end = triple['object']['offset'][1]
                try:
                    sub_start_, sub_end_ = align_char_span_text_b((sub_start, sub_end), offset_mapping) 
                    obj_start_, obj_end_ = align_char_span_text_b((obj_start,obj_end), offset_mapping)
                    so_ids[0][sub_start_][sub_end_-1] = 1
                    so_ids[1][obj_start_][obj_end_-1] = 1
                    head_ids[0][sub_start_][obj_start_] = 1
                    tail_ids[0][sub_end_-1][obj_end_-1] = 1
                except Exception as e:
                    log.warning(e)
                    log.warning(f'sub char offset {(sub_start, sub_end)} or obj char offset {(obj_start, obj_end)} align to token offset failed \n text: {text} \n mapping: {offset_mapping}')
                    continue
            batch_inputs['so_ids'].append(so_ids)
            batch_inputs['head_ids'].append(head_ids)
            batch_inputs['tail_ids'].append(tail_ids)
        batch_inputs['so_ids'] = torch.stack(batch_inputs['so_ids'], dim=0)
        batch_inputs['head_ids'] = torch.stack(batch_inputs['head_ids'], dim=0)
        batch_inputs['tail_ids'] = torch.stack(batch_inputs['tail_ids'], dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        return batch
        