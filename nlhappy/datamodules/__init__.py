from .text_classification import TextClassificationDataModule
from .text_pair import TextPairDataModule
from .token_classification import TokenClassificationDataModule
from .span_classification import SpanClassificationDataModule
from .triple_extraction import TripleExtractionDataModule


__all__ = [
    'TextClassificationDataModule', 
    'TextPairDataModule', 
    'TokenClassificationDataModule', 
    'SpanClassificationDataModule', 
    'TripleExtractionDataModule'
    ]