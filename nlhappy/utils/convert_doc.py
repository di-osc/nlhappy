from spacy.tokens import Doc
from datasets import Dataset 
from typing import List, Dict
from tqdm import tqdm 
import random




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




def convert_ents_to_prompt_span_dataset(docs: list,
                                       alias_dict: Dict[str, List]={}, 
                                       add_negtive_sample: bool = False,
                                       sentence_level: bool = False) -> Dataset:
    """将doc转换为prompt span数据集

    Args:
        docs (list): 待转换的文档列表
        alias_dict (Dict[str, List], optional): 同义词字典. Defaults to {}.
        add_negtive_sample (bool, optional): 是否添加负样本. Defaults to False.
        sentence_level (bool, optional): 是否转换为句子级别. Defaults to False.

    Returns:
        Dataset: 转换的数据集
    """
    all_labels = set([ent.label_ for doc in docs for ent in doc.ents])
    text_ls = []
    prompt_ls = []
    spans_ls = []
    for doc in tqdm(docs):
        if not sentence_level:
            label_dict = {}
            for ent in doc.ents:
                if ent.label_ not in label_dict:
                    label_dict[ent.label_] = []
                label_dict[ent.label_].append({'text': ent.text, 'offset':[ent.start_char, ent.end_char]})
            if len(label_dict) == 0:
                continue
            else:
                for label in label_dict:
                    text_ls.append(doc.text)
                    if label in alias_dict:
                        prompt_ls.append(random.choice(alias_dict[label]+[label]))
                    else: 
                        prompt_ls.append(label)
                    spans_ls.append(label_dict[label])
                if add_negtive_sample:
                    other_labels = all_labels - set(label_dict.keys())
                    if len(other_labels)>0:
                        text_ls.append(doc.text)
                        label = random.choice(list(other_labels))
                        if label in alias_dict:
                            prompt_ls.append(random.choice(alias_dict[label]+[label]))
                        else:
                            prompt_ls.append(label)
                        spans_ls.append([])
        else:
            for sent in doc.sents:
                label_dict = {}
                for ent in sent.ents:
                    if ent.label_ not in label_dict:
                        label_dict[ent.label_] = []
                    label_dict[ent.label_].append({'text': ent.text, 'offset':[ent.start_char, ent.end_char]})
                if len(label_dict) == 0:
                    continue
                else:
                    for label in label_dict:
                        text_ls.append(sent.text)
                        if label in alias_dict:
                            prompt_ls.append(random.choice(alias_dict[label]+[label]))
                        else: 
                            prompt_ls.append(label)
                        spans_ls.append(label_dict[label])
                    if add_negtive_sample:
                        other_labels = all_labels - set(label_dict.keys())
                        if len(other_labels)>0:
                            text_ls.append(sent.text)
                            label = random.choice(list(other_labels))
                            if label in alias_dict:
                                prompt_ls.append(random.choice(alias_dict[label]+[label]))
                            else:
                                prompt_ls.append(label)
                            spans_ls.append([])   
    print("保存数据....") 
    return Dataset.from_dict({'text': text_ls, 'prompt': prompt_ls, 'spans': spans_ls})