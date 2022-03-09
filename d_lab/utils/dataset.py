from datasets.features import Features
from datasets import Sequence, Value, Dataset 
from typing import List
from spacy.tokens import Doc

dataset_features = Features(
    {
    'overall': Sequence({'text': Value('string'), 'label': Value('string'), 'labels':Sequence(Value('string'))}), 
    'spans': Sequence({'offset':Sequence(Value('int8')), 'label': Value('string'), 'text': Value('string')}),
    'tokens': Sequence({'offset':Sequence(Value('int8')),'label':Value('string'), 'text': Value('string')})
        }
)


def convert_docs_to_dataset(docs: List[Doc]) -> Dataset:
    """
    Convert a document to a dataset.
    """
    d = {'overall': [], 'spans': [], 'tokens': []}
    for doc in docs:
        d['overall'].append({'text': doc.text, 'label': doc._.label, 'labels': doc._.labels})
        spans = []
        for span in doc.spans['all']:
            spans.append({'offset': (span.start_char, span.end_char), 'label': span.label_, 'text': span.text})
        d['spans'].append(spans)
        tokens = []
        for token in doc:
            bio = token.ent_iob_ + '-' + token.ent_type_ if token.ent_iob_ != 'O' else token.ent_iob_
            tokens.append({'offset': (token.idx, token.idx+1), 'label': bio, 'text': token.text})
        d['tokens'] .append(tokens)
    ds = Dataset
    ds.features = dataset_features
    ds = ds.from_dict(d)
    return ds


if __name__ == "__main__":
    import spacy
    from spacy.tokens import Span
    nlp = spacy.blank('zh')
    doc = nlp(u'我是一个中国人')
    doc.spans['all'] = [Span(doc, 0, 1, label='PER'), Span(doc, 2, 3, label='PER'), Span(doc, 4, 5, label='PER')]
    doc.set_ents([Span(doc, 2, 3, label='PER'), Span(doc, 4, 5, label='PER')])
    ds = convert_docs_to_dataset([doc])
    print(ds)