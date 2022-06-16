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