from datasets.features import Features
from datasets import DatasetDict, Sequence, Value, Dataset, concatenate_datasets 
from typing import List, Type
from spacy.tokens import Doc
from .data import Triple
from tqdm import tqdm
from spacy.language import Language
from spacy.training import Example 
from spacy.scorer import get_ner_prf

dataset_features = Features(
    {
    'text': Value('string'),  
    'labels':Sequence(Value('string')),
    'spans': Sequence({'offset':Sequence(Value('int8')), 'label': Value('string'), 'text': Value('string')}),
    'tokens': Sequence({'offset':Sequence(Value('int8')),'label':Value('string'), 'text': Value('string')}),
    'triples': Sequence({'subject': {'offset':Sequence(Value('int8')), 'text':Value('string')}, 'predicate': Value('string'), 'object': {'offset':Sequence(Value('int8')), 'text':Value('string')}})
        }
)


def convert_docs_to_dataset(docs: List[Doc], sentence_level: bool =False) -> Dataset:
    """
    Convert a document to a dataset.
    args:
        docs: a list of spacy.tokens.Doc
        sentence_level: whether to convert to sentence level dataset
    """

    if sentence_level:
        print('注意: 转换为句子级别数据集, 仅适用于token, span分类任务')
    if isinstance(docs, Doc):
        docs = [docs]
        

    d = {'text':[], 'labels':[],  'spans':[], 'tokens':[], 'triples':[]}
    for doc in tqdm(docs, desc='处理数据....'):
        if not sentence_level:
            d['text'].append(doc.text)
            d['labels'].append([label for label in doc._.labels])

            spans = []
            if ('all' in doc.spans) and len(doc.spans['all']) > 0:
                for span in doc.spans['all']:
                    spans.append({'offset': (span.start_char, span.end_char), 'label': span.label_, 'text': span.text})
            d['spans'].append(spans)

            tokens = []
            if len(doc.ents) > 0:
                for token in doc:
                    t = {'offset': (token.idx, token.idx+1), 'text': token.text}
                    bio = token.ent_iob_ + '-' + token.ent_type_ if token.ent_iob_ != 'O' else token.ent_iob_
                    t['label'] = bio
                    tokens.append(t)
            d['tokens'].append(tokens)

            triples = []
            for spo in doc._.spoes:
                sub = spo.subject
                pred = spo.predicate
                obj = spo.object
                triples.append({'subject': {'offset':(sub.start_char, sub.end_char), 'text':sub.text}, 'predicate': pred, 'object': {'offset':(obj.start_char, obj.end_char), 'text':obj.text}})
            d['triples'].append(triples)

        else:
            for sent in doc.sents:
                d['text'].append(sent.text)
                d['labels'].append([])
                sent_start = sent.start_char
                spans = []
                if ('all' in doc.spans) and len(doc.spans['all']) > 0:
                    for span in doc.spans['all']:
                        if span.sent == sent:
                            spans.append({'offset': (span.start_char - sent_start, span.end_char - sent_start), 'label': span.label_, 'text': span.text})
                d['spans'].append(spans)

                tokens = []
                if len(doc.ents) > 0:
                    for token in sent:
                        t = {'offset': (token.idx - sent_start, token.idx - sent_start+1), 'text': token.text}
                        bio = token.ent_iob_ + '-' + token.ent_type_ if token.ent_iob_ != 'O' else token.ent_iob_
                        t['label'] = bio
                        tokens.append(t)
                d['tokens'].append(tokens)

                triples = []
                d['triples'].append(triples)
    print("保存数据....")
    ds = Dataset.from_dict(d)
    return ds



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
    # if plot: 
    #     fig = plt.figure()
    #     types = type_dict.keys()
    #     nums = type_dict.values()
    #     s = len(types) // 5
    #     ax = fig.add_axes([0,0,s,s])
    #     ax.bar(range(len(types)), nums, width=0.5, label='num')
    #     plt.xticks(range(len(types)), types)
    #     plt.show()
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


    

    