import pickle
from typing import Dict, List
from ..algorithms.text_match import BM25
from spacy.tokens import Doc
from collections import defaultdict
from spacy.lang.zh import Chinese
from spacy.language import Language
import os
from ..models import BERTCrossEncoder
from ..utils.utils import get_logger

log = get_logger()

class EntityNormalizer:
    def __init__(self,
                nlp,
                name,
                norm_labels: List,
                topk: int =20,
                positive_label: str='1',
                strategy: str='pre',
                threshold: float = 0.5,
                device:str = 'cpu'
                ) -> None:
        super().__init__()
        self.name = name
        self.positive_label = positive_label
        self.topk = topk
        self.strategy = strategy
        self.threshold = threshold
        self.norm_labels = norm_labels
        self.device = device
    
    def init_match(self,
                    model_or_path):
        """初始化匹配模型

        Args:
            model_or_path: 
        """
        if isinstance(model_or_path, BERTCrossEncoder):
            self.match_model = model_or_path
            self.match_model.freeze()
            self.match_model.to(self.device)
            
        else:
            self.match_model= BERTCrossEncoder.load_from_checkpoint(model_or_path)
            self.match_model.freeze()
            self.match_model.to(self.device)
        
        
    def init_recall(self,
                    norm_names: List[str],
                    synonym_dict: Dict[str, set]={},
                    tokenizer= None,
                    is_retrain_docs:bool=True,
                    k1: float= 1.5,
                    b: float=0.75,
                    epsilon: float=0.25):
        """初始化bm25模型
        Args:
            norm_names (List[str]): 标准实体列表
            synonym_dict (Dict[str, set], optional): 同义词字典key为别名, value为标准词. Defaults to {}.
            tokenizer (_type_, optional): 分词器,如果为None则为字符切分. Defaults to None.
            is_retrain_docs (bool, optional): 是否保存文档. Defaults to True.
            k1 (float, optional): 词频缩放系数. Defaults to 1.5.
            b (float, optional): 文档长度影响系数,0-1取值,0为完全不影响. Defaults to 0.75.
            epsilon (float, optional): idf的下限值. Defaults to 0.25.
        """
        
        self.map_dict = defaultdict(set)
        for ent in norm_names:
            self.map_dict[ent].add(ent)
        for k, values in synonym_dict.items():
            for v in values:
                if k != v:
                    if k in norm_names:
                        self.map_dict[v].add(k)
                    elif v in norm_names:
                        self.map_dict[k].add(v)
        corpus = list(self.map_dict.keys())       
        self.recall_model = BM25(corpus=corpus,
                                k1=k1,
                                b=b,
                                epsilon=epsilon,
                                tokenizer=tokenizer,
                                is_retain_docs=is_retrain_docs)
            
        
    def normalize(self, query):
        recalls = self.recall_model.recall(query, topk=self.topk) #recalls: [['阿-基综合征', 21.098753076625275], ['阿基综合症', 19.811872460858027]]
        recall_names = [recall[0] for recall in recalls]
        if query in recall_names:
            if len(self.map_dict[query]) == 1:
                return list(self.map_dict[query])[0]
        result_ls = []
        recall_ls = []
        if self.strategy == 'pre':
            recall_ls = [recall[0] for recall in recalls]
            for recall in recall_ls:
                pred, score = self.match_model.predict(text_pair=[query, recall], device=self.device)[0]
                if pred == self.positive_label and score > self.threshold:
                    result_ls.append(recall)
                    break
            if len(result_ls) == 0:
                return ''
            return self.map_dict[result_ls[0]]
        elif self.strategy == 'post': 
            for r in  recalls:
                for v in self.map_dict[r[0]]:
                    if v not in recall_ls:
                        recall_ls.append(v)
            for recall in recall_ls:
                pred, score = self.match_model.predict(text_pair=[query, recall], device=self.device)[0]
                if pred == self.positive_label and score > self.threshold:
                    result_ls.append(recall)
                    break
            if len(result_ls) == 0:
                return ''
            return result_ls[0]
            
    def __call__(self, doc: Doc):
        for ent in doc.ents:
            if ent.label_ in self.norm_labels:
                norm_ = self.normalize(ent.text)
                if type(norm_) == set:
                    assert len(norm_) == 1, f'bad case {norm_}'
                    norm_ = list(norm_)[0]
                ent._.norm_name = norm_
        return doc
    
    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        # path : save_path/entity_normalizer
        if not os.path.exists(path):
            os.mkdir(path=path)
        match_path = os.path.join(path, 'match')
        if not os.path.exists(match_path):
            os.mkdir(match_path)
        match_model_path = os.path.join(match_path, 'bert_tm.pkl')
        with open(match_model_path, 'wb') as f:
            pickle.dump(self.match_model, f)
        recall_path = os.path.join(path, 'recall')
        if not os.path.exists(recall_path):
            os.mkdir(recall_path)
        map_dict = os.path.join(recall_path, 'map_dict.pkl')
        with open(map_dict, 'wb') as f:
            pickle.dump(self.map_dict, f)
        bm25_file = os.path.join(recall_path, 'bm25.pkl')
        with open(bm25_file, 'wb') as f:
            pickle.dump(self.recall_model, f)
        
    def from_disk(self, path:str, exclude):
        # path: load_path/entity_normalizer
        match_model_path = os.path.join(path, 'match', 'bert_tm.pkl')
        with open(match_model_path, 'rb') as f:
            self.match_model = pickle.load(f)
        self.match_model.freeze()
        try:
            self.match_model.to(self.device)
        except:
            log.info(f' to device {self.device} failed')
        recall_path = os.path.join(path, 'recall')
        map_dict_path = os.path.join(recall_path, 'map_dict.pkl')
        self.map_dict = pickle.load(open(map_dict_path, 'rb'))
        bm25_path = os.path.join(recall_path, 'bm25.pkl')
        self.recall_model = pickle.load(open(bm25_path,'rb'))
        

default_config = {
    'topk': 20,
    'positive_label': '1',
    'strategy': 'post',
    'threshold': 0.5,
    'device': 'cpu'
    }
        
        
@Chinese.factory('entity_normalizer',assigns=['span._.norm_name'],default_config=default_config)
def make_entity_normalizer(nlp: Language,
                            name: str,
                            norm_labels: List,
                            topk: int =20,
                            positive_label: str='1',
                            strategy: str='pre',
                            threshold: float = 0.5,
                            device: str = 'cpu'):
    
    return EntityNormalizer(nlp=nlp,
                            name=name,
                            norm_labels=norm_labels,
                            topk=topk,
                            positive_label=positive_label,
                            strategy=strategy,
                            threshold=threshold,
                            device=device)

        
        