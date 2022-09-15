from functools import lru_cache
import torch
from ..utils.make_datamodule import PLMBaseDataModule


class TokenClassificationDataModule(PLMBaseDataModule):
    '''序列标注数据模块
    数据集字段:
    - tokens: list[str], 例如: ['我', '是', '一', '中', '国', '人', '。']
    - labels: list[str], 例如: ['O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']
    '''
    def __init__(
        self,
        dataset: str,
        plm: str,
        max_length: int,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        pretrained_dir: str ,
        data_dir: str ,
        label_pad_id: int = -100
        ):
        super().__init__()

        self.save_hyperparameters()


    def transform(self, examples):
        batch = {'inputs':[], 'label_ids':[]}
        for i, t_list in enumerate(examples['tokens']):
            tokens = [t['text'] for t in t_list]
            label_ids = [self.hparams['label2id'][t['label']] for t in t_list]
            inputs = self.tokenizer.encode_plus(
                tokens,  
                add_special_tokens=True,
                padding='max_length',  
                max_length=self.hparams.max_length,
                truncation=True)
            inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
            if len(label_ids) < (self.hparams.max_length-2):
                O_id = self.hparams.label2id['O']
                label_ids = [O_id] + label_ids + [O_id] + [-100]*(self.hparams.max_length-len(label_ids)-2)
            else: label_ids = [O_id] + label_ids + [O_id]
            batch['inputs'].append(inputs)
            label_ids = torch.tensor(label_ids)
            batch['label_ids'].append(label_ids)
        return batch

    
    @property
    @lru_cache()
    def label2id(self):
        set_labels = sorted(set([t['label'] for t_list in self.dataset['train']['tokens'] for t in t_list]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        return label2id


    def setup(self, stage):
        self.hparams.label2id = self.label2id
        self.hparams.id2label = {i:l for l,i in self.label2id.items()}
        self.dataset.set_transform(transform=self.transform)
