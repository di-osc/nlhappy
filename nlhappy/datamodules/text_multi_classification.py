from typing import Any
from datasets import load_from_disk
from ..utils.make_datamodule import PLMBaseDataModule

class TextMultiClassification(PLMBaseDataModule):
    """多标签多分类的数据模块, 每个标签下有多重选择的情况"""
    def __init__(self,
                dataset: str,
                plm: str,
                max_length: int,
                batch_size: int,
                pin_memory: bool=False,
                num_workers: int=0,
                data_dir: str ='./datasets',
                plm_dir: str = './plms'):
        """多标签多分类数据模块, 每个标签下游多重选择

        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            max_length (int): 单文本最大长度
            batch_size (int): 批次大小
            pin_memory (bool, optional): _description_. Defaults to False.
            num_workers (int, optional): _description_. Defaults to 0.
            data_dir (str, optional): _description_. Defaults to './datasets'.
            plm_dir (str, optional): _description_. Defaults to './plms'.
        """
        super().__init__()
        self.save_hyperparameters()


    def transform(self, batch):
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        """需要设置参数label2id, id2label, token2id, bert_config最后要对dataset设置transform"""
        raise NotImplementedError
    
    @property
    def dataset(self):
        return load_from_disk(self.hparams.data_dir + self.hparams.dataset)