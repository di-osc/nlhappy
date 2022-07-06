from typing import Dict
import os
from transformers import AutoTokenizer

def get_hf_tokenizer(config: Dict, vocab: Dict):
    with open('./vocab.txt', 'w') as f:
        for k in vocab.keys():
            f.writelines(k + '\n')
    config.to_json_file('./config.json')
    tokenizer = AutoTokenizer.from_pretrained('./')
    os.remove('./vocab.txt')
    os.remove('./config.json')
    return tokenizer

def align_token_span(token_span_offset, token_offset_mapping):
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