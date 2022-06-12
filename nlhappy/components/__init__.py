from .span_classifier import make_spancat
from .token_classifier import make_tokencat
from .text_classifier import make_text_classification
from .sentence_segmenter import make_sentence_segmenter
from .triple_extractor import make_triple_extractor
from .event_extractor import make_event_extractor


def get_chinese_nlp():
    import spacy
    from .tokenizer import CharTokenizer
    nlp = spacy.blank('zh')
    nlp.tokenizer = CharTokenizer(nlp.vocab)
    return nlp