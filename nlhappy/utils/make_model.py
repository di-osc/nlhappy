from functools import lru_cache
from typing import Dict, Tuple, Union, List
import os
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertModel, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from lightning.pytorch import LightningModule
import torch
import tempfile
import json
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from datasets import load_from_disk, load_dataset
from torch.optim.lr_scheduler import CyclicLR


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
            d = json.dumps(dict(config))
            f.write(d)
        tokenizer = AutoTokenizer.from_pretrained(tmpdirname)
    return tokenizer


def get_hf_config_object(config: Union[DictConfig , Dict]) -> AutoConfig:
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
            d = json.dumps(dict(config))
            f.write(d)
        config = AutoConfig.from_pretrained(tmpdirname)
    return config

    
def align_token_span(token_span_offset: Tuple, token_offset_mapping: List[Tuple]) -> Tuple:
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
    
class BaseModel(LightningModule):
    """最基础的模型类,配置了
    """
    scheduler_names = ['linear_warmup', 'cosine_warmup', 'harmonic', 'cycle']
    
    def get_linear_warmup_step_scheduler_config(self, optimizer) -> Dict:
        total_steps = self.get_total_steps()
        warmup_steps = self.get_one_epoch_steps() // 3
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
    
    def get_cosine_warmup_step_scheduler_config(self, optimizer) -> Dict:
        total_steps = self.get_total_steps()
        warmup_steps = self.get_one_epoch_steps() // 3
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
    
    def get_harmonic_epoch_scheduler_config(self, optimizer):
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        scheduler_config = {'scheduler': scheduler, 'interval':'epoch'}
        return scheduler_config
    
    def get_cycle_step_scheduler_config(self, optimizer):
        step_size_up = self.get_one_epoch_steps() // 2
        max_lr = self.hparams.lr
        base_lr = max_lr / 4
        scheduler = CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=False)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
        
    def get_total_steps(self):
        return self.trainer.estimated_stepping_batches
    
    def get_one_epoch_steps(self):
        total_steps = self.get_total_steps()
        max_epochs = self.trainer.max_epochs
        return total_steps // max_epochs
    
    def get_scheduler_config(self, optimizer, name: str):
        """

        Args:
            optimizer (): 优化器实例
            name (str): 'harmonic_epoch', 'linear_warmup_step', 'cosine_warmup_step'之一

        Returns:
            dict: scheduler配置字典
        """
        if name == 'harmonic':
            return self.get_harmonic_epoch_scheduler_config(optimizer=optimizer)
        elif name == 'linear_warmup':
            return self.get_linear_warmup_step_scheduler_config(optimizer=optimizer)
        elif name == 'cosine_warmup':
            return self.get_cosine_warmup_step_scheduler_config(optimizer=optimizer)
        elif name == 'cycle':
            return self.get_cycle_step_scheduler_config(optimizer=optimizer)
    
    
    
class PLMBaseModel(LightningModule):
    """基于预训练语言模型的基类,在继承类的时候需要传入plm和plm_dir
    
    - 内置了scheduler,可以通过cls.scheduler_names查看所有的scheduler,通过self.get_scheduler_config方法得到pl的scheduler config
    - 通过self.tokenizer直接调用tokenizer
    - 通过self.get_plm_architecture可以得到没有加载参数的预训练模型的架构
    """
    
    scheduler_names = ['linear_warmup', 'cosine_warmup', 'harmonic', 'cycle']
    
    def __init__(self) -> None:
        super().__init__()
        # 保存所有参数
        self.save_hyperparameters()
        assert self.hparams.scheduler in self.scheduler_names, f'availabel names {self.scheduler_names}'
        assert 'plm' in self.hparams and 'plm_dir' in self.hparams, 'you have to at least pass in plm and plm_dir'
          
    @property
    @lru_cache()
    def tokenizer(self):
        keys = self.hparams.keys()
        if 'trf_config' in keys and 'vocab' in keys:
            try:
                tokenizer = get_hf_tokenizer(self.hparams.trf_config, self.hparams.vocab)
            except:
                tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.hparams.plm_dir, self.hparams.plm))
            return tokenizer
        elif 'plm' in keys and 'plm_dir' in keys:
            plm_path = Path(self.hparams.plm_dir, self.hparams.plm)
            if plm_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(plm_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.hparams.plm)
            return tokenizer

    @property
    @lru_cache()
    def trf_config(self) -> AutoConfig:
        if 'trf_config' in self.hparams.keys():
            return get_hf_config_object(self.hparams.trf_config)
        elif 'plm' in self.hparams.keys() and 'plm_dir' in self.hparams.keys():
            plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
            plm_config = AutoConfig.from_pretrained(plm_path) 
            self.hparams.trf_config = plm_config.to_dict() 
            self.hparams.vocab = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))  
            return plm_config
    
    def get_plm_architecture(self, add_pooler_layer: bool = False) -> torch.nn.Module:
        trf_config = self.trf_config
        trf_config.add_pooler_layer = add_pooler_layer
        return AutoModel.from_config(trf_config)    
    
    def get_linear_warmup_step_scheduler_config(self, optimizer) -> Dict:
        total_steps = self.get_total_steps()
        warmup_steps = self.get_one_epoch_steps() // 3
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
    
    def get_cosine_warmup_step_scheduler_config(self, optimizer) -> Dict:
        total_steps = self.get_total_steps()
        warmup_steps = self.get_one_epoch_steps() // 3
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
    
    def get_harmonic_epoch_scheduler_config(self, optimizer):
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        scheduler_config = {'scheduler': scheduler, 'interval':'epoch'}
        return scheduler_config
    
    def get_cycle_step_scheduler_config(self, optimizer):
        step_size_up = self.get_one_epoch_steps() // 2
        max_lr = self.hparams.lr
        base_lr = max_lr / 4
        scheduler = CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=False)
        scheduler_config = {'scheduler': scheduler, 'interval':'step'}
        return scheduler_config
        
    def get_total_steps(self):
        return self.trainer.estimated_stepping_batches
    
    def get_one_epoch_steps(self):
        total_steps = self.get_total_steps()
        max_epochs = self.trainer.max_epochs
        return total_steps // max_epochs
        
    def get_scheduler_config(self, optimizer, name: str):
        """

        Args:
            optimizer (): 优化器实例
            name (str): 'harmonic_epoch', 'linear_warmup_step', 'cosine_warmup_step'之一

        Returns:
            dict: scheduler配置字典
        """
        if name == 'harmonic':
            return self.get_harmonic_epoch_scheduler_config(optimizer=optimizer)
        elif name == 'linear_warmup':
            return self.get_linear_warmup_step_scheduler_config(optimizer=optimizer)
        elif name == 'cosine_warmup':
            return self.get_cosine_warmup_step_scheduler_config(optimizer=optimizer)
        elif name == 'cycle':
            return self.get_cycle_step_scheduler_config(optimizer=optimizer)

    def to_onnx(self, 
                file_path: str, 
                text: str = '中国人'):
        torch_inputs = self.tokenizer(text, return_tensors='pt')
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
        

class HFPretrainedModel(BaseModel):
    """配置了所有功能的模型基类
    """
    
    def __init__(self,
                 plm_max_length: int = 512,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last_batch: bool = False,
                 shuffle_train: bool = False,
                 shuffle_val: bool = False,
                 shuffle_test: bool = False):
        super().__init__()
        # 保存所有参数
        self.save_hyperparameters()
        assert 'plm' in self.hparams, 'you have to at least pass in plm'
        
    @property
    @lru_cache()
    def dataset(self):
        path = Path(self.hparams.dataset)
        if path.exists():
            return load_from_disk(path)
        else:
            return load_dataset(path=self.hparams.dataset)
          
    @property
    @lru_cache()
    def tokenizer(self):
        keys = self.hparams.keys()
        if 'trf_config' in keys and 'vocab' in keys:
            return get_hf_tokenizer(self.hparams.trf_config, self.hparams.vocab)
        elif 'plm' in keys:
            plm_path = Path(self.hparams.plm)
            tokenizer = AutoTokenizer.from_pretrained(plm_path)
            return tokenizer

    def get_plm_config(self):
        if 'trf_config' in self.hparams.keys():
            return get_hf_config_object(self.hparams.trf_config)
        elif 'plm' in self.hparams.keys():
            plm_config = AutoConfig.from_pretrained(self.hparams.plm)    
            return plm_config
        
    def get_hf_config_object(self, config: Union[DictConfig , Dict]) -> AutoConfig:
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
                d = json.dumps(dict(config))
                f.write(d)
            config = AutoConfig.from_pretrained(tmpdirname)
        return config
    
    def get_plm(self) -> torch.nn.Module:
        """获取预训练模型,用于初始化模型
        """
        if 'trf_config' in self.hparams.keys():
            plm_config = self.get_hf_config_object(self.hparams.trf_config)
            plm = AutoModel(plm_config)
        elif 'plm' in self.hparams.keys():
            plm = AutoModel.from_pretrained(self.hparams.plm)
            self.hparams.trf_config = plm.config.to_dict()
            self.hparams.vocab = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        return plm   

    def to_onnx(self, 
                file_path: str, 
                text: str = '中国人'):
        torch_inputs = self.tokenizer(text, return_tensors='pt')
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
        
    
    @property
    @lru_cache()
    def train_df(self):
        return self.dataset['train'].to_pandas()
    
    @property
    @lru_cache()
    def val_df(self):
        return self.dataset['validation'].to_pandas()
    
    @property
    @lru_cache()
    def test_df(self):
        return self.dataset['test'].to_pandas()
    
    def train_dataloader(self):
        return DataLoader(dataset= self.dataset['train'], 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_train,
                          batch_size=None,
                          sampler=BatchSampler(RandomSampler(self.dataset['train']), batch_size=self.hparams.batch_size, drop_last=self.hparams.drop_last_batch))
    
    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['validation'], 
                          batch_size=None, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_val,
                          sampler=BatchSampler(RandomSampler(self.dataset['validation']), batch_size=self.hparams.batch_size, drop_last=self.hparams.drop_last_batch))

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], 
                          batch_size=None, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_test,
                          sampler=BatchSampler(RandomSampler(self.dataset['test']), batch_size=self.hparams.batch_size, drop_last=self.hparams.drop_last_batch))



class HFBertModel(BaseModel):
    """配置了所有功能的模型基类
    """
    
    def __init__(self,
                 plm_dir: str,
                 input_max_length: int = 512):
        super().__init__()
        # 保存所有参数
        self.save_hyperparameters()
        self.input_max_length = input_max_length
        self.bert = self.get_plm()
        
    def on_train_start(self) -> None:
        ckpt_path = Path(self.hparams.plm_dir, 'pytorch_model.bin')
        if ckpt_path.exists():
            self.bert.load_state_dict(torch.load(ckpt_path))
        else:
            self.bert = self.bert.from_pretrained(self.hparams.plm_dir)
        
    @property
    @lru_cache()
    def tokenizer(self):
        keys = self.hparams.keys()
        if 'trf_config' in keys and 'vocab' in keys:
            return get_hf_tokenizer(self.hparams.trf_config, self.hparams.vocab)
        elif 'plm' in keys:
            plm_path = Path(self.hparams.plm_dir)
            tokenizer = AutoTokenizer.from_pretrained(plm_path)
            return tokenizer

    def get_plm_config(self):
        if 'trf_config' in self.hparams.keys():
            return get_hf_config_object(self.hparams.trf_config)
        elif 'plm' in self.hparams.keys():
            plm_config = BertConfig.from_pretrained(self.hparams.plm_dir)    
            return plm_config
        
    def get_hf_config_object(self, config: Union[DictConfig , Dict]) -> AutoConfig:
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
                d = json.dumps(dict(config))
                f.write(d)
            config = AutoConfig.from_pretrained(tmpdirname)
        return config
    
    def get_plm(self) -> torch.nn.Module:
        """获取预训练模型,用于初始化模型
        """
        if 'trf_config' in self.hparams.keys():
            plm_config = self.get_hf_config_object(self.hparams.trf_config)
            plm = BertModel(plm_config)
        elif 'plm' in self.hparams.keys():
            plm = BertModel.from_pretrained(self.hparams.plm_dir)
            self.hparams.trf_config = plm.config.to_dict()
            self.hparams.vocab = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        return plm   

    def to_onnx(self, 
                file_path: str, 
                text: str = '中国人'):
        torch_inputs = self.tokenizer(text, return_tensors='pt')
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