import spacy
from typing import Any
from spacy.tokens import Doc
import unicodedata



class CharTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def __call__(self, text) -> Any:
        words = [s for s in text]
        spaces = [False] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
    def to_disk(self, path):
        pass
    
    def from_disk(self, path):
        pass 
    

    
        
class BasicTokenizer(object):
    """基础分词器
    - 中文按照字切分
    - 英文按照词切分
    - 数字单个切分
    - 标点切分
    - 英文转小写,去掉重音符号
    - """

    def __init__(self,
                 do_lower_case=False,
                 return_offset=False,
                 never_split=[]):
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.return_offset = return_offset

    def tokenize(self, text):
        """文本切分成token"""
        origin = text
        text = self._clean_text(origin)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = self._whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = self._whitespace_tokenize(" ".join(split_tokens))
        if self.return_offset:
            offsets = self.rematch(text=origin, tokens=output_tokens)
            return output_tokens, offsets
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text:str):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp) or char.isdigit():
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _whitespace_tokenize(self, text):
        """去除文本中的空白符"""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens



    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if len(char) == 0:
            return False
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False


    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if len(char) ==0:
            return False
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False


    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def _lowercase_and_normalize(self, text):
        """转小写，并进行简单的标准化
        """
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        return text

    def _is_special(self, ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """

        if self.do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self.do_lower_case:
                ch = self._lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append(())
            else:
                if self.do_lower_case:
                    token = token.lower()
                start = text[offset:].index(token) + offset
                end = start + len(token)
                # token_mapping.append(char_mapping[start:end])
                token_mapping.append((start, end))
                offset = end

        return token_mapping
    
    def __call__(self, text: str) -> Any:
        tokens = self.tokenize(text)
        spaces = [False] * len(tokens)
        return Doc(self.vocab, words=tokens, spaces=spaces)
    
    def to_disk(self, path):
        pass
    
    def from_disk(self, path):
        pass
    
    
all_tokenizers = {'char':CharTokenizer, 
                 'basic':BasicTokenizer}