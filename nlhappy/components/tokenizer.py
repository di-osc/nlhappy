import spacy
from typing import Any
from spacy.tokens import Doc


class CharTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def __call__(self, text) -> Any:
        words = [s for s in text]
        spaces = [False] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)