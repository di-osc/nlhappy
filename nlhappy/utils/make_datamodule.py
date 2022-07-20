import os
import oss2
import zipfile
from .utils import get_logger
from typing import List

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


default_access_key_id = 'LTAI5t6MP68LoaghwWStqxuC'
default_access_key_secret = 'v0DSdRawIXcZnVCeq0eZ1cldAy5DQ1'
default_endpoint = 'http://oss-cn-beijing.aliyuncs.com'
default_data_bucket = 'deepset'
default_model_bucket = 'pretrained-model'

class OSSStorer:
    '''阿里云oss对象存储'''
    def __init__(
        self, 
        access_key_id : str = default_access_key_id,
        access_key_secret : str = default_access_key_secret, 
        endpoint :str = default_endpoint, 
        data_bucket : str = default_data_bucket,
        model_bucket : str = default_model_bucket,
        ):
        super().__init__()
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.data_bucket = oss2.Bucket(self.auth, endpoint, data_bucket)
        self.model_bucket = oss2.Bucket(self.auth, endpoint, model_bucket)


    def download_dataset(
        self, 
        dataset:str, 
        localpath: str='./datasets/'):
        """下载数据集
        - dataset: 数据集名称
        - localpath: 下载到本地的路径 默认为./datasets/
        """
        if not os.path.exists(localpath):
            os.makedirs(localpath)
        file = dataset + '.zip'
        file_path = localpath + file
        dataset_path = localpath + dataset
        if not os.path.exists(dataset_path):
            try:
                self.data_bucket.get_object_to_file(key=file, filename=file_path)
                with zipfile.ZipFile(file=file_path, mode='r') as zf:
                    zf.extractall(path=localpath)
            finally:
                if os.path.exists(file_path):
                    os.remove(path=file_path)


    def download_plm(
        self, 
        model:str, 
        localpath: str = './plms/'):
        """下载预训练模型
        - model: 模型名称
        - localpath: 下载到本地的路径 默认为./plms/
        """
        if not os.path.exists(localpath):
            os.makedirs(localpath)
        file = model + '.zip'
        file_path = localpath + file
        model_path = localpath + model
        if not os.path.exists(model_path):
            try:
                self.model_bucket.get_object_to_file(key=file, filename=file_path)
                with zipfile.ZipFile(file=file_path, mode='r') as zf:
                    zf.extractall(path=localpath)
            finally:
                if os.path.exists(file_path):
                    os.remove(path=file_path)
                

log = get_logger(__name__)  
        
def prepare_data_from_oss(dataset: str,
                          plm: str,
                          dataset_dir: str ='./datasets/',
                          plm_dir: str = './plms/') -> None:
        '''
        下载数据集.这个方法只会在一个GPU上执行一次.
        '''
        oss = OSSStorer()
        dataset_path = os.path.join(dataset_dir, dataset)
        plm_path = os.path.join(plm_dir, plm)
        # 检测数据
        log.info('Checking all data ...' )
        if os.path.exists(dataset_path):
            log.info(f'{dataset_path} already exists.')
        else:
            log.info('not exists dataset in {}'.format(dataset_path))
            log.info('start downloading dataset from oss')
            oss.download_dataset(dataset, dataset_dir)
            log.info('finish downloading dataset from oss')
        if os.path.exists(plm_path):
            log.info(f'{plm_path} already exists.') 
        else : 
            log.info('not exists plm in {}'.format(plm_path))
            log.info('start downloading plm from oss')
            oss.download_plm(plm, plm_dir)
            log.info('finish downloading plm from oss')
        log.info('all data are ready')
        
        
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
    


