from functools import lru_cache
from typing import Dict, Union
import os
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from pytorch_lightning import LightningModule
import torch
from typing import Any
import tempfile
import json
from omegaconf import OmegaConf, DictConfig


def get_hf_tokenizer(config: Union[Dict, DictConfig] , vocab: Union[Dict, DictConfig]):
    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_path = os.path.join(tmpdirname, 'vocab.txt')
        config_path = os.path.join(tmpdirname, 'config.json')
        with open(vocab_path, 'w') as f:
            for k in vocab.keys():
                f.writelines(k + '\n')
        with open(config_path, 'w') as f:
            if type(config) == DictConfig:
                config = OmegaConf.to_container(config)
            assert type(config)==dict, f'config must be type dict, but found {type(config)}'
            d = json.dumps(config)
            f.write(d)
        tokenizer = AutoTokenizer.from_pretrained(tmpdirname)
    return tokenizer

def get_hf_config_object(config: Union[DictConfig , Dict]):
    with tempfile.TemporaryDirectory() as tmpdirname:
        config_path = os.path.join(tmpdirname, 'config.json')
        with open(config_path, 'w') as f:
            if type(config) == DictConfig:
                config = OmegaConf.to_container(config)
            try:
                config = config.to_dict() # 兼容huffingface config 对象
            except:
                pass
            assert type(config)==dict, f'config must be type dict, but found {type(config)}'
            d = json.dumps(config)
            f.write(d)
        config = AutoConfig.from_pretrained(tmpdirname)
    return config

    

def align_token_span(token_span_offset, token_offset_mapping):
    '''将词符级别的下标对齐为字符级别的下标
    参数
    - token_span_offset: 例如(0, 1) 下标指的是字符的下标
    - token_offset_mapping: 每个词符与字符对应的下标[(0,1),(1,2)]
    返回
    char_span_offset: (0,2)
    '''
    char_span_offset = ()
    if token_span_offset[1] - token_span_offset[0] == 1:
        char_span_offset = token_offset_mapping[token_span_offset[0]]
        return char_span_offset

    else:
        start = token_offset_mapping[token_span_offset[0]][0]
        end = token_offset_mapping[token_span_offset[1]-1][1]
        char_span_offset = (start, end)
        return char_span_offset
    
    
class PLMBaseModel(LightningModule):
    """基于预训练语音模型的基类
    内置功能:
    - 
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        if 'label2id' not in self.hparams and 'id2label' in self.hparams:
            self.hparams.label2id = {l:i for i,l in self.hparams.id2label.items()}
        
    
    @property
    @lru_cache()
    def tokenizer(self):
        keys = self.hparams.keys()
        if 'trf_config' in keys and 'vocab' in keys:
            return get_hf_tokenizer(self.hparams.trf_config, self.hparams.vocab)
        elif 'plm' in keys and 'plm_dir' in keys:
            plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
            return AutoTokenizer.from_pretrained(plm_path)
    
    
    def get_plm_architecture(self):
        if 'trf_config' in self.hparams.keys():
            plm_config = get_hf_config_object(self.hparams.trf_config)
            return AutoModel.from_config(plm_config)
        elif 'plm' in self.hparams.keys() and 'plm_dir' in self.hparams.keys():
            plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
            plm_config = AutoConfig.from_pretrained(plm_path)
            self.hparams.trf_config = plm_config.to_dict()
            self.hparams.vocab = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
            return AutoModel.from_config(plm_config)
    
    
    def get_linear_warmup_step_scheduler_config(self, optimizer, warmup_rate: float = 0.5):
        steps_per_epoch = self.get_one_epoch_steps()
        total_steps = self.trainer.max_epochs * steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_rate * steps_per_epoch)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
    
    
    def get_cosine_warmup_step_scheduler_config(self, optimizer, warmup_rate: float = 0.5):
        steps_per_epoch = self.get_one_epoch_steps()
        total_steps = self.trainer.max_epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_rate * steps_per_epoch)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
    
    
    def get_harmonic_epoch_scheduler_config(self, optimizer):
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        scheduler_config = {'scheduler': scheduler, 'interval':'epoch'}
        return scheduler_config
    
    
    def get_one_epoch_steps(self):
        if type(self.trainer.gpus) == int:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // self.trainer.gpus
        if type(self.trainer.gpus) == list:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // len(self.trainer.gpus)
        return steps_per_epoch
    
    
    def get_scheduler_config(self, optimizer, name: str):
        """

        Args:
            optimizer (): 优化器实例
            name (str): 'harmonic_epoch', 'linear_warmup_step', 'cosine_warmup_step'之一

        Returns:
            dict: scheduler配置字典
        """
        availabel_names = ['harmonic_epoch', 'linear_warmup_step', 'cosine_warmup_step']
        assert name in availabel_names, f'availabel names {availabel_names}'
        if name == 'harmonic_epoch':
            return self.get_harmonic_epoch_scheduler_config(optimizer)
        elif name == 'linear_warmup_step':
            return self.get_linear_warmup_step_scheduler_config(optimizer=optimizer)
        elif name == 'cosine_warmup_step':
            return self.get_cosine_warmup_step_scheduler_config(optimizer=optimizer)

    
    def to_onnx(self, 
                file_path: str, 
                text_a: str = '中国人', 
                text_b : str = '中国'):
        torch_inputs = self.tokenizer(text_a, text_b, return_tensors='pt')
        dynamic_axes = {
                    'input_ids': {0: 'batch', 1: 'seq'},
                    'attention_mask': {0: 'batch', 1: 'seq'},
                    'token_type_ids': {0: 'batch', 1: 'seq'},
                }
        with torch.no_grad():
            torch.onnx.export(model=self,
                              args=tuple(torch_inputs.values()), 
                              f=file_path, 
                              input_names=list(torch_inputs.keys()),
                              dynamic_axes=dynamic_axes, 
                              opset_version=14,
                              output_names=['logits'],
                              export_params=True)
        print('export to onnx successfully')