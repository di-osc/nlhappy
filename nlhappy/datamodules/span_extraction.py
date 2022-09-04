from ..utils.make_datamodule import PLMBaseDataModule, char_idx_to_token
import torch
from functools import lru_cache


class SpanExtractionDataModule(PLMBaseDataModule):
    """span分类的数据模块 数据集必须有text, spans两个字段
    """
    def __init__(self,
                dataset: str,
                plm: str,
                batch_size: int,
                transform: str = 'globalpointer',
                auto_length: str='max',
                **kwargs):
        super().__init__()
        self.transforms['globalpointer'] = self.gp_transform

    @property
    @lru_cache()
    def label2id(self):
        set_labels = sorted(set([span['label'] for spans in self.dataset['train']['spans'] for span in spans]))
        label2id = {label: i for i, label in enumerate(set_labels)}
        return label2id


    def gp_transform(self, example):
        batch_text = example['text']
        batch_spans = example['spans']
        max_length = self.hparams.max_length
        batch_inputs = {'input_ids':[],'token_type_ids':[],'attention_mask':[]}
        batch_span_ids = []
        for i, text in enumerate(batch_text):
            inputs = self.tokenizer(text, 
                                    padding='max_length',  
                                    max_length=max_length,
                                    truncation=True,
                                    return_offsets_mapping=True)
            spans = batch_spans[i]
            mapping = inputs['offset_mapping']
            span_ids = torch.zeros(len(self.hparams.label2id), max_length, max_length)
            for span in spans :
                # +1 是因为添加了 [CLS]
                start = span['offset'][0]
                start = char_idx_to_token(start, mapping)
                end = span['offset'][1] 
                end = char_idx_to_token(end, mapping)
                label_id = self.hparams.label2id[span['label']]
                span_ids[label_id,  start, end] = 1
            batch_inputs['input_ids'].append(inputs['input_ids'])
            batch_inputs['token_type_ids'].append(inputs['token_type_ids'])
            batch_inputs['attention_mask'].append(inputs['attention_mask'])
            batch_span_ids.append(span_ids)
        batch_span_ids = torch.stack(batch_span_ids, dim=0)
        batch = dict(zip(batch_inputs.keys(), map(torch.tensor, batch_inputs.values())))
        batch['span_ids'] = batch_span_ids
        return batch


    def setup(self, stage: str) -> None:
        self.hparams.max_length = self.get_max_length()
        self.hparams.label2id = self.label2id
        self.dataset.set_transform(transform=self.transforms.get(self.hparams.transform))

    @staticmethod
    def show_one_sample(self):
        return {'text':'我的电话是12345','spans':[{'label':'电话','offset':[5, 10], 'text':'12345'}]}


    