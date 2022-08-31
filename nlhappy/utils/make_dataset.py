from datasets import Dataset, DatasetDict
from typing import Dict, Tuple, Union, List
from ..algorithms.text_match import BM25
from spacy.tokens import Doc
import random
from tqdm import tqdm




def train_val_split(dataset: Dataset, 
                    val_frac: float =0.1,
                    return_dataset_dict: bool =True) -> Union[Tuple[Dataset, Dataset], DatasetDict]:
    """split dataset into tarin and validation datasets

    Args:
        dataset (Dataset): dataset to split
        val_frac (float, optional): validation radio of all dataset. Defaults to 0.1.
        return_dataset_dict (bool, optional): if return_dataset_dict is True, return a DatasetDict,
            otherwise return a tuple of train, val datasets. Defaults to True.

    Returns:
        Union[Tuple[Dataset, Dataset], DatasetDict]: if return_dataset_dict is True, return a DatasetDict, otherwise return a tuple of train, val datasets
    """
    df = dataset.to_pandas()
    train_df = df.sample(frac=1-val_frac)
    val_df = df.drop(train_df.index)
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    if not return_dataset_dict:
        return train_ds, val_ds
    else:
        return DatasetDict({'train': train_ds, 'validation': val_ds})

def train_val_test_split(dataset: Dataset,
                         val_frac: float =0.1,
                         test_frac: float =0.1,
                         return_dataset_dict: bool =True) -> Union[Tuple[Dataset, Dataset, Dataset], DatasetDict]:
    """split dataset into tarin vlidation and test datasets

    Args:
        dataset (Dataset): dataset to split
        val_frac (float, optional): validation radio of all dataset. Defaults to 0.1.
        test_frac (float, optional): test radio of all dataset. Defaults to 0.1.

    Returns:
        Union[Tuple[Dataset, Dataset, Dataset], DatasetDict]: if return_dataset_dict is True, return a DatasetDict, 
            otherwise return a tuple of train, val, test datasets
        
    """
    df = dataset.to_pandas()
    train_df = df.sample(frac=1-val_frac-test_frac)
    other_df = df.drop(train_df.index)
    val_df = other_df.sample(frac=1-(test_frac/(val_frac+test_frac)))
    test_df = other_df.drop(val_df.index)
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    if not return_dataset_dict:
        return train_ds, val_ds, test_ds
    else:
        return DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})
        

def make_text_match_dataset_with_bm25(corpus: List[str],
                                      synonym_dict: Dict[str, set],
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
    

def make_re_dataset_from_docs(docs: List[Doc], 
                              prompt_type: bool=False,
                              num_negative: int = 2): 
    """将doc._.relations 转换为relation extraction任务数据

    Args:
        docs (List[Doc]): 待转换docs
        prompt_type (bool, optional): 是否为prompt形式. Defaults to False.

    Returns:
        Dataset: all convert dataset
        
    Example:
        dataset: 
        {'text': '小张毕业于河北工程大学',
        'triples': [{'object': {'offset': [5,11], 'text': '河北工程大学'},
                    'predicate': '毕业院校',
                    'subject': {'offset': [0, 2], 'text': '小张'}}]},
    """
    if not prompt_type:
        text_ls = []
        triple_ls = []
        for doc in tqdm(docs):
            text_ls.append(doc.text)
            triples = []
            for rel in doc._.relations:
                sub = rel.sub
                pred = rel.label
                for obj in rel.objs:
                    triples.append({'subject':{'offset':(sub.start_char, sub.end_char), 'text':sub.text},
                                    'predicate': pred,
                                    'object':{'offset':(obj.start_char, obj.end_char), 'text':obj.text}})
            triple_ls.append(triples)
        ds = Dataset.from_dict({'text':text_ls, 'triples':triple_ls})
        return ds
    else:
        text_ls = []
        triple_ls = []
        prompt_ls = []
        all_labels = set([rel.label for doc in docs for rel in doc._.relations])
        for doc in tqdm(docs):
            triple_dict = {}
            for rel in doc._.relations:
                sub = rel.sub
                p = rel.label
                if p not in triple_dict:
                    triple_dict[p] = []
                for obj in rel.objs:
                    triple_dict[p].append({'subject':{'offset':(sub.start_char, sub.end_char), 'text':sub.text},
                                    'object':{'offset':(obj.start_char, obj.end_char), 'text':obj.text}})
            
            for p in triple_dict:
                text_ls.append(doc.text)
                triple_ls.append(triple_dict[p])
                prompt_ls.append(p)
            other_labels = list(all_labels - triple_dict.keys())
            for l in other_labels[:num_negative]:
                text_ls.append(doc.text)
                triple_ls.append([])
                prompt_ls.append(l)
                
        ds = Dataset.from_dict({'text':text_ls, 'triples':triple_ls, 'prompts': prompt_ls})
        return ds


def make_ee_dataset_from_doc(docs: List[Doc]):
    text_ls = []
    event_ls = []
    for doc in tqdm(docs):
        text_ls.append(doc.text)
        events = []
        for event in doc._.events:
            e = {'label':event.label, 'roles':[]}
            for role in event.roles:
                spans = event.roles[role]
                for span in spans:
                    e['roles'].append({'label':role, 
                                    'offset':(span.start_char, span.end_char), 
                                    'text':span.text})
            events.append(e)
        event_ls.append(events)
    ds = Dataset.from_dict({'text':text_ls, 'events':event_ls})
    return ds
        
            
        


    
    
    
    