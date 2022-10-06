import os
from .utils import get_logger
from typing import List, Optional, Dict, Union
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
import pytorch_lightning as pl
from transformers import AutoConfig, AutoTokenizer, AutoModel
from functools import lru_cache


log = get_logger()

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


def prepare_data_from_huffingface(dataset: str,
                             plm: str,
                             dataset_dir: str ='./datasets/',
                             plm_dir: str = './plms/') -> None:
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        dataset_path = os.path.join(dataset_dir, dataset)
        plm_path = os.path.join(plm_dir, plm)
        # 检测数据
        if os.path.exists(dataset_path):
            pass
        else:
            log.info('cannot found dataset in {}.'.format(dataset_path))
            try:
                log.info(f'download dataset {dataset} from huffingface')
                dataset = load_dataset(dataset)
                dataset.save_to_disk(dataset_path)
                log.info(f'download dataset succeed')
            except:
                log.error('download dataset failed')

        if os.path.exists(plm_path):
            pass 
        else : 
            log.info('cannot found plm in {}'.format(plm_path))
            try:
                log.info(f'download plm {plm} from huffingface')
                model = AutoModel.from_pretrained(plm)
                tokenizer = AutoTokenizer.from_pretrained(plm)
                model.save_pretrained(plm_path)
                tokenizer.save_pretrained(plm_path)
                log.info(f'download plm succeed')
            except:
                log.error('download plm failed')



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
                 plm_max_input_length: int = 512,
                 dataset_dir: str = 'datasets',
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 shuffle_train: bool = True,
                 shuffle_val: bool = False,
                 shuffle_test: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.transforms = {}
    
    
    def prepare_data(self) -> None:
        prepare_data_from_huffingface(dataset=self.hparams.dataset,
                                      plm=self.hparams.plm,
                                      dataset_dir=self.hparams.dataset_dir,
                                      plm_dir=self.hparams.plm_dir)
    
    
    @property
    @lru_cache()
    def dataset(self):
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        dsd = load_from_disk(dataset_path)
        return dsd
            
            
    @lru_cache()
    def get_trf_config(self):
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        config = AutoConfig.from_pretrained(plm_path)
        config = config.to_dict()
        return config
    
    
    @property
    @lru_cache()
    def tokenizer(self):
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        return AutoTokenizer.from_pretrained(plm_path)
    
    
    def get_vocab(self):
        return dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))


    def get_available_transforms(self):
        return self.transforms.keys()
    
        
    @lru_cache()
    def get_max_length(self, set_to_hparams: bool = True):
        """根据auto_length参数自动获取最大token的长度,并将其添加进hparams
        Args:
            set_to_hparam (bool): 调用该方法时自动设置为self.hparams.max_length,默认为True

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
        max_length_ = min(self.hparams.plm_max_input_length, max_length+2)
        log.info(f'current max token length: {max_length_}')
        if set_to_hparams:
            self.hparams.max_length = max_length_
        return max_length_
    
    
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
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_train)
    
    
    def val_dataloader(self):
        return DataLoader(dataset=self.dataset['validation'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_val)


    def test_dataloader(self):
        return DataLoader(dataset=self.dataset['test'], 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=self.hparams.pin_memory,
                          shuffle=self.hparams.shuffle_test)