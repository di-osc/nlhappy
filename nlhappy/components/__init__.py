from .span_classifier import make_spancat
from .token_classifier import make_tokencat
from .text_classifier import make_text_classification
from .sentence_segmenter import make_sentence_segmenter
from .triple_extractor import make_triple_extractor
from .tokenizer import CharTokenizer
import spacy



def make_zh_nlp(tokenize_type: str = 'char'):
    nlp = spacy.blank('zh')
    if tokenize_type == 'char':
        nlp.tokenizer = CharTokenizer(vocab=nlp.vocab)
    return nlp
