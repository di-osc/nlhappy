from spacy.tokens import Doc
from typing import List
from spacy.training import Example 
from spacy.scorer import get_ner_prf, Scorer



def is_ent_badcase(pred: Doc, true: Doc):
    """判断两个doc是否相等"""
    assert pred.text == true.text
    if len(pred.ents) != len(true.ents):
        return True
    else: 
        pred_offsets = set([(ent.start, ent.end, ent.label_) for ent in pred.ents])
        true_offsets = set([(ent.start, ent.end, ent.label_) for ent in true.ents])
        return len(pred_offsets - true_offsets) != 0 


def analysis_ent_type(docs: List[Doc], return_ent_per_type: bool=True, ent_lt:int =100):
    """分析得到各个实体类型所对应的数量
    参数:
    - docs: doc列表
    - return_ent_per_type: 是否返回实体类型数量对应字典,默认返回
    - return_ent_lt: 数量, 返回小于这个数量的实体的doc
    """
    
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
    - return_prf: 是否返回每个实体类别的prf值
    """
    badcases = []
    for pred, doc  in zip(preds, docs):
        if is_ent_badcase(pred, doc):
            badcases.append((doc, pred))
    if return_prf:
        examples = [Example(pred, true) for (pred, true) in zip(preds, docs)]
        prf = get_ner_prf(examples=examples)
        return badcases, prf
    else: return badcases
    

def analysis_relation_badcase(examples: List[Example] , return_prf:bool=True):
    """错误分析,并且返回细粒度的指标

    Args:
        examples (List[Example]): 一个列表,里面为Example
        return_prf (bool, optional): 是否返回prf分数. Defaults to True.

    Returns:
        List[Examples], Dict : badcase, prf分数
    """
    all_preds = 0
    all_corrects = 0
    all_trues = 0
    per_types = {rel.label : {'preds':0, 'trues':0, 'corrects':0} for e in examples for rel in e.y._.relations}
    badcases = []
    for e in examples:
        corrects = [ rel for rel in e.x._.relations if rel in e.y._.relations]
        if len(corrects) != len(e.y._.relations):
            badcases.append(e)
        if return_prf:
            all_trues += len(e.y._.relations)
            all_preds += len(e.x._.relations)
            all_corrects += len(corrects)
            for rel in e.y._.relations:
                per_types[rel.label]['trues'] +=1
            for rel in e.x._.relations:
                per_types[rel.label]['preds'] +=1
                if rel in e.y._.relations:
                    per_types[rel.label]['corrects'] +=1
    if return_prf:
        rel_p = all_corrects/(all_preds + 1e-8)
        rel_r = all_corrects/(all_trues + 1e-8)
        rel_f = 2*rel_p*rel_r/(rel_p + rel_r + 1e-8)
        rel_per_types = {l:{} for l in per_types}
        for l, scores in per_types.items():
            p = scores['corrects'] / (scores['preds'] + 1e-8)
            r = scores['corrects'] / (scores['trues'] + 1e-8)
            f = 2 * p * r / (p + r + 1e-8)
            rel_per_types[l]['p'] = p
            rel_per_types[l]['r'] = r
            rel_per_types[l]['f'] = f
            
        scores = {'rel_micro_p':rel_p, 
                'rel_micro_r': rel_r,
                'rel_micro_f': rel_f,
                'rel_per_types':rel_per_types}
        return badcases, scores
    else:
        return badcases
    
def analysis_text_badcase(examples: List[Example], return_prf: bool=True):
    all_preds = 0
    all_corrects = 0
    all_trues = 0
    per_types = {e.y._.label : {'preds':0, 'trues':0, 'corrects':0} for e in examples}
    badcases = []
    for e in examples:
        if e.x._.label == e.y._.label and e.x._.label != '':
            all_corrects += 1
        else:
            badcases.append(e)
        if return_prf:
            all_trues += 1
            all_preds += 1
            per_types[e.y._.label]['trues'] +=1
            if e.x._.label != '':
                per_types[e.x._.label]['preds'] +=1
                if e.x._.label == e.y._.label:
                    per_types[e.x._.label]['corrects'] +=1
    if return_prf:
        rel_p = all_corrects/(all_preds + 1e-8)
        rel_r = all_corrects/(all_trues + 1e-8)
        rel_f = 2*rel_p*rel_r/(rel_p + rel_r + 1e-8)
        rel_per_types = {l:{} for l in per_types}
        for l, scores in per_types.items():
            p = scores['corrects'] / (scores['preds'] + 1e-8)
            r = scores['corrects'] / (scores['trues'] + 1e-8)
            f = 2 * p * r / (p + r + 1e-8)
            rel_per_types[l]['p'] = p
            rel_per_types[l]['r'] = r
            rel_per_types[l]['f'] = f
            
        scores = {'text_micro_p':rel_p, 
                'text_micro_r': rel_r,
                'text_micro_f': rel_f,
                'text_per_types':rel_per_types}
        return badcases, scores
    else:
        return badcases
    
            
        
        
    