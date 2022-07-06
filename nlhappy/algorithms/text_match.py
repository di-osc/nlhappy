import math
import copy
import logging
import numpy as np
from six import iteritems

logger = logging.getLogger(__name__)



class BM25(object):
    """
    BM25模型

    Args:
        corpus (:obj:`list`):
            检索的语料
        k1 (:obj:`float`, optional, defaults to 1.5):
            取正值的调优参数，用于文档中的词项频率进行缩放控制
        b (:obj:`float`, optional, defaults to 0.75):
            0到1之间的参数，决定文档长度的缩放程度，b=1表示基于文档长度对词项权重进行完全的缩放，b=0表示归一化时不考虑文档长度因素
        epsilon (:obj:`float`, optional, defaults to 0.25):
            idf的下限值
        tokenizer (:obj:`object`, optional, defaults to None):
            分词器，用于对文档进行分词操作，默认为None，按字颗粒对文档进行分词
        is_retain_docs (:obj:`bool`, optional, defaults to True):
            是否保持原始文档

    Reference:
        [1] https://github.com/RaRe-Technologies/gensim/blob/3.8.3/gensim/summarization/bm25.py
    """  # noqa: ignore flake8"

    def __init__(
        self,
        corpus,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        tokenizer=None,
        is_retain_docs=True
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.docs = None
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if is_retain_docs:
            self.docs = copy.deepcopy(corpus)

        if tokenizer:
            self.tokenizer = tokenizer
            corpus = [self.tokenizer.tokenize(document) for document in corpus]
        else:
            corpus = [list(document) for document in corpus]

        self._initialize(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:                        
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size

        idf_sum = 0
        negative_idfs = []
        for word, freq in iteritems(nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        if self.average_idf < 0:
            logger.warning(
                'Average inverse document frequency is less than zero. Your corpus of {} documents'
                ' is either too small or it does not originate from natural text. BM25 may produce'
                ' unintuitive results.'.format(self.corpus_size)
            )

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_score(self, query, index):
        score = 0.0
        doc_freqs = self.doc_freqs[index]
        numerator_constant = self.k1 + 1
        denominator_constant = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
        for word in query:
            if word in doc_freqs:
                df = self.doc_freqs[index][word]
                idf = self.idf[word]
                score += (idf * df * numerator_constant) / (df + denominator_constant)
        return score

    def get_scores(self, query):
        scores = [self.get_score(query, index) for index in range(self.corpus_size)]
        return scores

    def recall(self, query: str, topk=5):
        if self.tokenizer:
            query = self.tokenizer.tokenize(query)
        else: query = [s for s in query]
        scores = self.get_scores(query)
        indexs = np.argsort(scores)[::-1][:topk]

        if self.docs is None:
            return [[i, scores[i]] for i in indexs]
        else:
            return [[self.docs[i], scores[i]] for i in indexs]
        
        
        