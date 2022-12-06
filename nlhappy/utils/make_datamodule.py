import os
from .utils import get_logger
from typing import Union, List
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from datasets import load_from_disk, load_dataset, DatasetDict
import pytorch_lightning as pl
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedTokenizerFast
from functools import lru_cache
from pathlib import Path
import numpy as np
from pytorch_lightning import LightningDataModule


log = get_logger()


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def char_idx_to_token(char_idx, offset_mapping):
    """
    将char级别的idx 转换为token级别的idx
    例如:
    text = '我是中国人'
    tokens = ['我', '是', '中国', '人']
    '国'的字符下标为3, token级别的下标为2
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= char_idx < span[1]:
            return index
    return -1
        
        
def align_char_span(char_span_offset: tuple, 
                    token_offset_mapping, 
                    special_offset=(0,0),
                    add_special: bool=False) -> tuple:
    '''对齐字符级别的span标签与bert的子词切分的标签
    参数:
    - char_span_offsets: 字符级别的文本片段下标,例如(0,1)
    - token_offset_mapping: 词符与字符下标映射例如[(0,1), (1,2), (2,3) ...]
    输出
    token_span_offset: 词符级别的文本片段下标,例如(0,2)
    '''
    token_span_offset = ()
    bias = 0
    for i, offset in enumerate(token_offset_mapping):
        if offset == special_offset:
            bias += 1
            continue
        if offset != special_offset:
            if offset[0] == char_span_offset[0]:
                start = i
                if offset[1] == char_span_offset[1]:
                    end = i+1
                    break
                else: 
                    continue
            if offset[1] == char_span_offset[1]:
                end = i+1
                break
    try:
        if add_special:
            token_span_offset = (start, end)
        else:
            token_span_offset = (start-bias, end-bias)     
    except:
        log.warning('align offset failed')
    return token_span_offset

def align_char_span_text_b(char_span_offset: tuple, 
                            token_offset_mapping, 
                            special_offset=(0,0),
                            add_pre: bool=True) -> tuple:
    """对齐文本b的token下标, 适用于bert text pair形式

    Args:
        char_span_offset (tuple): 文本中的char级别的offset
        token_offset_mapping (_type_): 映射字典
        special_offset (tuple, optional): 特殊token offset. Defaults to (0,0).
        add_pre (bool, optional): 是否加上之前的token. Defaults to True.

    Returns:
        tuple: token级别的offset
    """
    token_span_offset = ()
    bias = 0
    for i, offset in enumerate(token_offset_mapping):
        if offset == special_offset and i ==0:
            continue
        if offset == special_offset:
            bias = i+1
        if offset != special_offset and bias>0:
            if offset[0] == char_span_offset[0]:
                start = i
                if offset[1] == char_span_offset[1]:
                    end = i+1
                    break
                else: 
                    continue
            if offset[1] == char_span_offset[1]:
                end = i+1
                break
    try:
        if add_pre:
            token_span_offset = (start, end)
        else:
            token_span_offset = (start-bias, end-bias)     
    except:
        log.warning(f'align {char_span_offset} failed')
    return token_span_offset


def prepare_dataset(dataset_name: str, dataset_dir) -> None:
    path = Path(dataset_dir, dataset_name)
    if path.exists():
        pass
    else:
        log.info('cannot found dataset in {}.'.format(path))
        try:
            log.info(f'download dataset {dataset_name} from huffingface')
            dataset = load_dataset(dataset_name)
            log.info(f'download dataset succeed')
        except Exception as e:
            log.error('download dataset failed')
            raise(e)
            
            
def prepare_plm(plm_name: str, plm_dir: str) -> None:
    path = Path(plm_dir, plm_name)
    if path.exists():
        pass
    else : 
        log.info('cannot found plm in {}'.format(path))
        try:
            log.info(f'download plm {plm_name} from huffingface')
            model = AutoModel.from_pretrained(plm_name)
            tokenizer = AutoTokenizer.from_pretrained(plm_name)
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            log.info(f'download plm succeed')
        except:
            log.error('download plm failed')
        


def prepare_from_huffingface(dataset: str, 
                             dataset_dir: str,
                             plm: str,
                             plm_dir: str) -> None:
        '''
        下载数据集和预训练模型这个方法只会在一个GPU上执行一次.
        '''
        prepare_dataset(dataset_name=dataset, dataset_dir=dataset_dir)
        prepare_plm(plm_name=plm, plm_dir=plm_dir)


class PLMBaseDataModule(pl.LightningModule):
    """数据模块的基类,子类需要完成setup方法,子类初始化的时候至少包含dataset,plm,batch_size参数,
    内置功能:
    - 自动保存超参数
    - 不同策略自动获取最大文本长度,超过512则取512
    - 下载数据集和预训练模型
    - 自动读取tokenizer
    - 自动读取数据集
    - 自动设置dataloader,数据集需要切分为train,validation,test
    """
    def __init__(self,
                 auto_length: Union[str, int] = 'max',
                 plm_dir: str = 'plms',
                 plm_max_length: int = 512,
                 dataset_dir: str = 'datasets',
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 shuffle_train: bool = False,
                 shuffle_val: bool = False,
                 shuffle_test: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.transforms = {}
    
    
    def prepare_data(self) -> None:
        prepare_from_huffingface(dataset=self.hparams.dataset, 
                                 dataset_dir=self.hparams.dataset_dir,
                                 plm=self.hparams.plm,
                                 plm_dir=self.hparams.plm_dir)
    
    @property
    @lru_cache()
    def dataset(self) -> DatasetDict:
        dataset_path = Path(self.hparams.dataset_dir, self.hparams.dataset)
        if dataset_path.exists():  
            dsd = load_from_disk(dataset_path)
        else:
            dsd = load_dataset(self.hparams.dataset)
        return dsd
            
            
    @lru_cache()
    def get_trf_config(self):
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        config = AutoConfig.from_pretrained(plm_path)
        config = config.to_dict()
        return config
    
    
    @property
    @lru_cache()
    def tokenizer(self) -> PreTrainedTokenizerFast:
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        return AutoTokenizer.from_pretrained(plm_path)
    
    
    def get_vocab(self):
        return dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))


    def get_available_transforms(self):
        return self.transforms.keys()
    
        
    @lru_cache()
    def get_max_length(self):
        """根据auto_length参数自动获取最大token的长度
        
        Returns:
            int: 最大token长度
        """
        length = self.train_df.text.map(lambda x: len(self.tokenizer.tokenize(x)))
        if self.hparams.auto_length == 'max':
            max_length = length.max()
        if self.hparams.auto_length == 'mean':
            max_length = int(length.mean())
        if type(self.hparams.auto_length) == int:
            assert self.hparams.auto_length >0, 'max_length length  must > 0'
            max_length = self.hparams.auto_length
        return max_length
    
    
    def get_batch_max_length(self, batch_text: List[str]) -> int:
        """获取一个batch的最大token长度,不会大于预训练模型的最大输入长度,一般用于dataset的transform中

        Args:
            batch_text (List[str]): 一个批次的文本

        Returns:
            int: 最大文本长度
        """
        max_length = max([len(self.tokenizer.encode(t)) for t in batch_text]) # 获取最大token序列的长度
        max_length = min([self.hparams.plm_max_length, max_length])
        return max_length
    
    
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
                          sampler=BatchSampler(RandomSampler(self.dataset['train']), batch_size=self.hparams.batch_size, drop_last=False))
    
    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['validation'], 
                          batch_size=None, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_val,
                          sampler=BatchSampler(RandomSampler(self.dataset['validation']), batch_size=self.hparams.batch_size, drop_last=False))

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], 
                          batch_size=None, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_test,
                          sampler=BatchSampler(RandomSampler(self.dataset['test']), batch_size=self.hparams.batch_size, drop_last=False))


class BaseDataModule(LightningDataModule):
    def __init__(self,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 shuffle_train: bool = False,
                 shuffle_val: bool = False,
                 shuffle_test: bool = False,
                 drop_last: bool = False):
        super().__init__()
        self.save_hyperparameters()
        assert 'batch_size' in self.hparams and 'dataset_path' in self.hparams and 'tokenizer_path' in self.hparams, '子类至少需要传入dataset_path, tokenizer_path, batch_size参数'
        
    @property
    @lru_cache()
    def dataset(self):
        path = Path(self.hparams.dataset_path)
        if path.exists():
            return load_from_disk(path)
        else:
            return load_dataset(path=self.hparams.dataset_path)
        
    @property
    @lru_cache()
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.hparams.tokenizer_path)
        
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
                          sampler=BatchSampler(RandomSampler(self.dataset['train']), batch_size=self.hparams.batch_size, drop_last=self.hparams.drop_last))
    
    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['validation'], 
                          batch_size=None, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_val,
                          sampler=BatchSampler(RandomSampler(self.dataset['validation']), batch_size=self.hparams.batch_size, drop_last=self.hparams.drop_last))

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], 
                          batch_size=None, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_test,
                          sampler=BatchSampler(RandomSampler(self.dataset['test']), batch_size=self.hparams.batch_size, drop_last=self.hparams.drop_last))