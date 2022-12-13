from ..utils.make_datamodule import PLMBaseDataModule, sequence_padding
import torch


class QuestionAnsweringDataModule(PLMBaseDataModule):
    """span分类的数据模块 数据集必须有text, spans两个字段
    """
    def __init__(self,
                dataset: str,
                batch_size: int,
                plm: str = 'hfl/chinese-roberta-wwm-ext',
                **kwargs):
        super().__init__()
        
    def gp_transform(self, examples):
        batch_text = examples['text']
        batch_question = examples['question']
        batch_spans = examples['spans']
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_span_ids = []
        batch_inputs = self.tokenizer(batch_question,
                                      batch_text, 
                                      padding=True,  
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors='pt')
        for i, text in enumerate(batch_text):
            spans = batch_spans[i]
            span_ids = torch.zeros(1, max_length, max_length)
            for span in spans :
                start = span['indices'][0]
                _start = batch_inputs.char_to_token(i, start, 1)
                end = span['indices'][-1] 
                _end = batch_inputs.char_to_token(i, end, 1)
                span_ids[0, _start, _end] = 1
            batch_span_ids.append(span_ids)
        batch_span_ids = torch.stack(batch_span_ids, dim=0)
        batch_inputs['span_tags'] = batch_span_ids
        return batch_inputs
    
    def sequence_transform(self, examples):
        batch_text = examples['text']
        batch_question = examples['question']
        batch_spans = examples['spans']
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_tags = []
        batch_inputs = self.tokenizer(batch_question,
                                      batch_text, 
                                      padding=True,  
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors='pt')
        for i, text in enumerate(batch_text):
            spans = batch_spans[i]
            tags = torch.zeros(max_length, dtype=torch.long)
            for span in spans :
                for idx in span['indices']: 
                    token_idx = batch_inputs.char_to_token(i, idx, 1)
                    tags[token_idx] = 1
            batch_tags.append(tags)
        batch_tags = torch.stack(batch_tags)
        batch_inputs['tags'] = batch_tags
        return batch_inputs
    
    def pointer_transform(self, examples):
        batch_text = examples['text']
        batch_question = examples['question']
        batch_spans = examples['spans']
        max_length = self.get_batch_max_length(batch_text=batch_text)
        batch_start_tags = []
        batch_end_tags = []
        batch_inputs = self.tokenizer(batch_question,
                                      batch_text, 
                                      padding=True,  
                                      truncation=True,
                                      max_length=max_length,
                                      return_tensors='pt')
        for i, text in enumerate(batch_text):
            spans = batch_spans[i]
            start_tags = torch.zeros(max_length, dtype=torch.long)
            end_tags = torch.zeros(max_length, dtype=torch.long)
            for span in spans :
                for idx in span['indices'][0:]:
                    _start = batch_inputs.char_to_token(batch_or_char_index=i, char_index=idx, sequence_index=1)
                    if _start is not None:
                        break
                for idx in span['indices'][::-1]:
                    _end = batch_inputs.char_to_token(batch_or_char_index=i, char_index=idx, sequence_index=1)
                    if _end is not None:
                        break
                start_tags[_start] = 1
                end_tags[_end] = 1
            batch_start_tags.append(start_tags)
            batch_end_tags.append(end_tags)
        batch_start_tags = torch.stack(batch_start_tags)
        batch_end_tags = torch.stack(batch_end_tags)
        batch_inputs['start_tags'] = batch_start_tags
        batch_inputs['end_tags'] = batch_end_tags
        return batch_inputs