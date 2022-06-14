from spacy.tokens import Doc
from typing import List
from spacy.training import Example 
from spacy.scorer import get_ner_prf



def is_badcase(pred: Doc, true: Doc):
    """判断两个doc是否相等"""
    assert pred.text == true.text
    if len(true._.labels) >0:
        return pred._.labels != true._.labels
    elif len(true.ents) >0:
        return [ent.text for ent in true.ents] != [ent.text for ent in pred.ents]
    elif len(true.spans['all']) > 0:
        return true.spans['all'] != pred.spans['all']
    elif len(true._.triples) >0:
        return true._.triples != pred._.triples
    else: print("未能检查到标注数据")


def analysis_ent_type(docs: List[Doc], return_ent_per_type: bool=True, ent_lt:int =100):
    """分析得到各个实体类型所对应的数量
    参数:
    - docs: doc列表
    - return_ent_per_type: 是否返回实体类型数量对应字典,默认返回
    - return_ent_lt: 数量, 返回小于这个数量的实体的doc"""
    
    ent_types = set([ent.label_ for doc in docs for ent in doc.ents])
    type_dict = dict(zip(ent_types, [0]*len(ent_types)))
    for doc in docs:
        for ent in doc.ents:
            type_dict[ent.label_]+=1
    type_dict = {k:v for k,v  in sorted(type_dict.items(), key=lambda item: item[1])}
    return_types = [k for k,v in type_dict.items() if v < ent_lt]
    return_docs = {k:[] for k in return_types}
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ in return_types:
                return_docs[ent.label_].append(doc)
    if return_ent_per_type:
        return type_dict, return_docs
    else: return return_docs


def analysis_ent_badcase(preds: List[Doc], docs:List[Doc], return_prf:bool=False):
    """寻找badcase
    - preds: 预测的doc
    - docs: 真实的doc
    - return_prf: 是否返回每个实体类别的prf值"""
    badcases = []
    for pred, doc  in zip(preds, docs):
        if is_badcase(pred, doc):
            badcases.append((doc, pred))
    if return_prf:
        examples = [Example(pred, true) for (pred, true) in zip(preds, docs)]
        prf = get_ner_prf(examples=examples)
        return badcases, prf
    else: return badcases