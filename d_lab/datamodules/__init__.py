from .text_classification import TextClassificationDataModule
from .sentence_pair import SentencePairDataModule
from .token_classification import TokenClassificationDataModule
from .span_classification import SpanClassificationDataModule
from .triple_extraction import TripleExtractionDataModule


__all__ = [
    'TextClassificationDataModule', 
    'SentencePairDataModule', 
    'TokenClassificationDataModule', 
    'SpanClassificationDataModule', 
    'TripleExtractionDataModule'
    ]