from functools import lru_cache
from typing import Dict, Union
import os
from transformers import AutoTokenizer, AutoConfig, AutoModel
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
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        
    
    @property
    @lru_cache
    def tokenizer(self):
        return get_hf_tokenizer(self.hparams.trf_config, self.hparams.vocab)
    
    
    def get_plm_architecture(self):
        plm_config = get_hf_config_object(self.hparams.trf_config)
        return AutoModel.from_config(plm_config)

    
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