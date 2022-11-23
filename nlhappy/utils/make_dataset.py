from datasets import Dataset
from typing import Dict, List, Optional
from ..algorithms.text_match import BM25
import random
from tqdm import tqdm

        
def make_text_match_dataset_with_bm25(corpus: List[str],
                                     synonym_dict: Optional[Dict[str, set]] = None,
                                     num_positive_samples: int=10,
                                     num_negative_samples: int=20,
                                     recall_topk: int = 1000,
                                     reverse_sample: bool = False,
                                     positive_label: str = '1',
                                     negative_label: str='0',
                                     tokenizer = None,
                                     k1: float = 1.5,
                                     b: float=0.75,
                                     epsilon: float=0.25,
                                     is_retrain_docs=True,
                                     return_bm25: bool = False):
    """制作文本匹配二分类数据集, 此数据集适用于text_pair_classification任务

    Args:
        corpus (List[str]): 所有的可以召回的语料,词语等
        synonym_dict (Dict[str, set]): 标准词对应的同义词字典.字典的键应该包含于corpus.
        num_negative_samples (int): 负样本的数量.
        recall_topk(int): 召回模型的召回数量
        reverse_sample (bool): 是否逆序负样本采样
        positive_label (str): 正样本标签, 默认为'1'.
        negative_label (str): 负样本标签, 默认为'0'.
        tokenizer (_type_, optional): 分词器,默认为字符切分. Defaults to None.
        k1 (float, optional): bm25参数. Defaults to 1.5.
        b (float, optional): bm25参数. Defaults to 0.75.
        epsilon (float, optional): bm25参数. Defaults to 0.25.
        is_retrain_docs (bool, optional): bm25参数, 是否保留文档. Defaults to True.
        return_bm25 (bool): 是否返回bm25模型. Defaults to False.
    """
    bm25 = BM25(corpus=corpus, 
                k1=k1, 
                b=b, 
                epsilon=epsilon,
                is_retain_docs=is_retrain_docs,
                tokenizer=tokenizer)
    label_ls = []
    text_a_ls = []
    text_b_ls = []
    for key in tqdm(synonym_dict.keys()):
        recalls = bm25.recall(key, topk=recall_topk)
        recalls = [r[0] for r in recalls if r[0] != key and r[0] not in synonym_dict[key]]
        if reverse_sample:
            recalls.reverse()
            negatives = recalls[:num_negative_samples]
        else:
            negatives = recalls[:num_negative_samples]
        text_a_ls.extend(negatives)
        text_b_ls.extend([key]*len(negatives))
        label_ls.extend([negative_label]*len(negatives))
        syms = list(synonym_dict[key])
        positives = random.choices(syms, k=num_positive_samples)
        text_a_ls.extend(positives)
        text_b_ls.extend([key]*num_positive_samples)
        label_ls.extend([positive_label]*num_positive_samples)
    ds = Dataset.from_dict({'text_a':text_a_ls, 'text_b':text_b_ls, 'label': label_ls})
    if not return_bm25:
        return ds
    else:
        return ds, bm25