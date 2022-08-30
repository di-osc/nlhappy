from ..utils.make_datamodule import char_idx_to_token, PLMBaseDataModule
import torch


class PromptSpanExtractionDataModule(PLMBaseDataModule):
    """
    Data module for the prompt span extraction task.

    Dataset examples:
    {'text':'北京是中国的首都', 'prompt':'国家', 'spans':[{'text': '中国', 'offset':(3, 5)}]}
    {'text':'北京是中国的首都', 'prompt':'中国的首都', 'spans':[{'text': '北京', 'offset':(0, 2)}]}
    {'text': '北京是中国的首都', 'prompt': '北京的国家', 'spans': [{'text': '中国', 'offset': (3, 5)}]}
    其中offset 为左闭右开的字符级别下标
    """
    
    def __init__(self,
                 dataset: str,
                 plm: str,
                 transform: str,
                 batch_size: int,
                 max_length: int = -1,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 dataset_dir: str = 'datasets',
                 plm_dir: str = 'plms') :
        """基于模板提示的文本片段抽取数据模块

        Args:
            dataset (str): 数据集名称
            plm (str): 预训练模型名称
            max_length (int): 文本最大长度
            batch_size (int): 批次大小
        """
        super().__init__()
        
        self.transforms = {'global_span': self.global_span_transform}
        
        
    def setup(self, stage: str) -> None:
        self.dataset.set_transform(transform=self.transforms.get(self.hparams.transform))
        
 
    def global_span_transform(self, example):
        batch_text = example['text']
        batch_spans = example['spans']
        batch_prompt = example['prompt']
        max_length = self.hparams.max_length
        batch = {'inputs': [], 'span_ids': []}
        for i, text in enumerate(batch_text):
            # tokens = fine_grade_tokenize(text, self.tokenizer)
            prompt = batch_prompt[i]
            inputs = self.tokenizer(
                prompt, 
                text, 
                padding='max_length',  
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True)
            offset_mapping = [list(x) for x in inputs["offset_mapping"]]
            bias = 0
            for index in range(len(offset_mapping)):
                if index == 0:
                    continue
                mapping = offset_mapping[index]
                if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                    bias = index
                if mapping[0] == 0 and mapping[1] == 0:
                    continue
                offset_mapping[index][0] += bias
                offset_mapping[index][1] += bias
            spans = batch_spans[i]
            span_ids = torch.zeros(1, max_length, max_length)
            for span in spans:
                start = char_idx_to_token(span['offset'][0]+bias, offset_mapping)
                end = char_idx_to_token(span['offset'][1]-1+bias, offset_mapping)
                # 如果超出边界，则跳过
                try:
                    span_ids[0, start, end] = 1.0
                except Exception:
                    self.log.warning(f'set span {(start, end)} out of boudry')
                    pass 
            del inputs['offset_mapping']
            inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
            batch['inputs'].append(inputs)
            batch['span_ids'].append(span_ids)
        return batch
    
            

    
    
    