from typing import List
from datasets.features import Features
from datasets import DatasetDict, Sequence, Value, Dataset, concatenate_datasets 
from typing import List, Type
from spacy.tokens import Doc
from .data import Triple
from tqdm import tqdm

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def convert_tags_to_spans(seq: List[str]):
    """
    将bio或bios标注的序列转换为span形式(label, start, end).
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        convert_tags_to_spans(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


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
    for doc in tqdm(docs):
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

    ds = Dataset.from_dict(d)
    return ds