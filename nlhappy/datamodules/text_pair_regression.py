import torch
from transformers import AutoConfig, AutoTokenizer
from datasets import load_from_disk
import os
from ..utils.make_datamodule import PLMBaseDataModule


class TextPairRegressionDataModule(PLMBaseDataModule):
    '''文本对相似度数据模块
    dataset_exmaple:
        {'text_a': '左膝退变伴游离体','text_b': '单侧膝关节骨性关节病','similarity': 0}
    '''
    def __init__(self,
                dataset: str,
                plm: str,
                batch_size: int,
                transform: str):
        """
        Args:
            dataset (str): the name of the dataset.
            plm (str): the name of the plm.
            batch_size (int): the batch size in training and validation.
            transform (str): dataset transform
        """
        super().__init__()

        # 这一行代码为了保存传入的参数可以当做self.hparams的属性
        self.save_hyperparameters(logger=False)

        
    def setup(self, stage: str):
        dataset_path = os.path.join(self.hparams.dataset_dir, self.hparams.dataset)
        self.dataset = load_from_disk(dataset_path)
        plm_path = os.path.join(self.hparams.plm_dir, self.hparams.plm)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        self.hparams['vocab'] = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        self.hparams['trf_config'] = AutoConfig.from_pretrained(plm_path)
        self.dataset.set_transform(transform=self.transform)

    def transform(self, examples):
        batch_text_a = examples['text_a']
        batch_text_b = examples['text_b']
        similarities = examples['similarity']
        batch = {'inputs_a': [], 'inputs_b': [], 'similarities':[]}
        for i  in range(len(batch_text_a)):
            inputs_a= self.tokenizer(batch_text_a[i], 
                                    padding='max_length', 
                                    max_length=self.hparams.max_length, 
                                    truncation=True)
            inputs_a = dict(zip(inputs_a.keys(), map(torch.tensor, inputs_a.values())))
            batch['inputs_a'].append(inputs_a)
            inputs_b = self.tokenizer(batch_text_b[i],
                                    padding='max_length', 
                                    max_length=self.hparams.max_length, 
                                    truncation=True)
            inputs_b = dict(zip(inputs_b.keys(), map(torch.tensor, inputs_b.values())))
            batch['inputs_b'].append(inputs_b)
            batch['similarities'].append(torch.tensor(similarities[i], dtype=torch.float))
        
        return batch