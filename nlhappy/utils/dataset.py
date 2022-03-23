from datasets.features import Features
from datasets import Sequence, Value, Dataset 
from typing import List, Type
from spacy.tokens import Doc
from .data import Triple
from tqdm import tqdm

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


if __name__ == "__main__":
    import spacy
    from spacy.tokens import Span
    nlp = spacy.blank('zh')
    nlp.add_pipe('sentence_segmenter')
    doc = nlp(u'我是一个中  国人. 我是一个好人')
    doc.spans['all'] = [Span(doc, 4, 7, label='PER'), Span(doc, 12, 14, label='PER'), Span(doc, 4, 5, label='PER')]
    doc.set_ents([Span(doc, 2, 3, label='PER'), Span(doc, 4, 5, label='PER')])
    doc._.triples = [Triple((0, 1, 'PER', 2, 3)), Triple((2, 3, 'PER', 4, 5))]
    ds = convert_docs_to_dataset([doc], sentence_level=True)
    print(ds[0])
    print(ds[1])