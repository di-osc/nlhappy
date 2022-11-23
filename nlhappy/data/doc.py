from pydantic import BaseModel, conint, conint, constr, validator, conlist, validate_arguments
from typing import List, Optional, Union
import srsly
from pathlib import Path
import pandas as pd
from .dataset import Dataset
from ..utils.text import split_sentence


Label = constr(strip_whitespace=True, min_length=1)
Index = conint(ge=0, strict=True)
    
class Span(BaseModel):
    """原始文本的一个片段,可以连续也可以非连续
    参数:
    - text (str): 文本,默认None
    - is_continuous (bool): 是否连续,默认True
    - indices (List[int]): 对应文本的下标
    """
    text: constr(min_length=1) = None
    is_continuous: bool = True
    indices: List[Index]
    
    @validator('text')
    def validate_text(cls, v):
        if v:
            assert len(v.strip()) > 0, f'span的有效文本程度必须大于0'
        return v
    
    @validator('indices')
    def validate_indices(cls, v, values):
        if values['is_continuous']:
            for i in range(1, len(v)):
                assert v[i] - v[i-1] == 1, f'{v}实体非连续,如必要将is_continuous设置True'
        if values['text']:
            assert len(v) == len(values['text']), f'{values["text"]} 与 {v} 长度不一致'
        return v
    
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other: "Span") -> bool:
        if self.is_continuous and other.is_continuous:
            return self.indices[0] == other.indices[0] and self.indices[-1] == other.indices[-1]
        else:
            return self.indices == other.indices
    
    def __contains__(self, item: "Span"):
        return self.indices[0] <= item.indices[0] and self.indices[-1] >= item.indices[-1]

        


class Entity(Span):
    """有标签的Span,支持连续非连续类型
    参数:
    - text (str): 实体文本
    - label (str): 实体标签
    - is_continuous (bool): 是否连续,默认True
    - indices (List[int]): 字符级别下标
    """
    label: Label
        
    def __hash__(self):
        return hash(self.label)
    
    def __eq__(self, other: "Entity") -> bool:
        if self.is_continuous and other.is_continuous:
            return self.indices[0] == other.indices[0] and self.indices[-1] == other.indices[-1] and self.label == other.label
        else:
            return self.indices == other.indices and self.label == other.label
    
    
class Relation(BaseModel):
    """实体之间的有指向关系

    参数:
    - s(Entity): 主体
    - p(str): 关系标签
    - o(Entity): 客体
    """
    s: Entity
    p: Label
    o: Entity
    
    @validator('o')
    def validate_rel(cls, v, values):
        assert values['s'] != v, '主体客体同一实体'
        return v
    
    def __hash__(self):
        return hash(self.p)
    
    def __eq__(self, other: "Relation") -> bool:
        return self.s == other.s and self.p == other.p and self.o == other.o
    
    
class Event(BaseModel):
    """由若干实体和触发词组成的事件
    - args (List[Entity]): 组成事件的论元
    - label (Label): 事件类型标签
    - trigger(Optional[Span]): 触发词span,默认为None
    """
    
    args: conlist(item_type=Entity, min_items=1, unique_items=True)
    label: Label
    trigger: Optional[Span] = None
    
    def __hash__(self):
        return hash(self.label)
    
    def __eq__(self, other: "Event") -> bool:
        return self.args == other.args and self.label == other.label and self.trigger == other.trigger


class Doc(BaseModel):
    """存放所有标注数据,由模型或者原始数据构建
    参数:
    - text(str): 文本
    - label(str): 唯一标签
    - labels(List[str]): 多标签
    - ents(List[Entity]): 实体列表
    - rels(List[Relation]): 关系列表
    - events(List[Event]): 事件列表
    - sents(List(Span)): 文本连续的句子,不需要初始化,自动生成
    - summary(str): 摘要
    - title(str): 标题
    - id(str): 唯一标识符
    """
    
    text: constr(min_length=1)
    id: Optional[str] = None
    label: Optional[Label] = None
    labels: conlist(item_type=Entity, unique_items=True, min_items=1) = None
    ents: conlist(item_type=Entity, unique_items=True, min_items=1) = None
    rels: List[Relation] = None
    events: List[Event] = None   
    summary: constr(min_length=1, strip_whitespace=True, strict=True) = None
    title: constr(min_length=1, strip_whitespace=True, strict=True) = None
    
    @property
    def sents(self):
        sents = []
        for s in split_sentence(self.text):
            start = 0 if len(sents)==0 else sents[-1].indices[-1] + 1
            end = start + len(s)
            sents.append(Span(text=s, indices=[i for i in range(start, end)]))
        return sents
    
    @validator('text')
    def validate_text(cls, v: str):
        assert len(v.strip()) > 0, f'文本为空格'
        return v
    
    @validator('ents')
    def validate_ents(cls, v: List[Entity], values):
        if v:
            text = values['text']
            for ent in v:
                # 如果实体文本存在则确保实体文本与下标对应文本一致,确保标注正确
                if ent.text:
                    indices = ent.indices
                    ent_chars = [s for s in ent.text]
                    chars = [text[idx] for idx in indices]
                    assert ent_chars == chars, f'实体{ent_chars}与标注实体{chars}不一致'     
        return v
    
    @validator('rels')
    def validate_rels(cls, v: List[Relation], values):
        if v:
            text = values['text']
            for rel in v:
                s = rel.s
                o = rel.o
                if s.text:
                    indices = s.indices
                    s_chars = [s for s in s.text]
                    chars = [text[idx] for idx in indices]
                    assert s_chars == chars, f'主体{s_chars}与标注实体{chars}不一致'
                if o.text:
                    indices = o.indices
                    o_chars = [s for s in o.text]
                    chars = [text[idx] for idx in indices]
                    assert o_chars == chars, f'客体{o_chars}与标注实体{chars}不一致'
        return v
    
    @validator('events')
    def validate_events(cls, v: List[Event], values):
        if v:
            text = values['text']
            for e in v:
                for arg in e.args:
                    if arg.text:
                        indices = arg.indices
                        arg_chars = [s for s in arg.text]
                        chars = [text[idx] for idx in indices]
                        assert arg_chars == chars, f'事件论元{arg_chars}与标注论元{chars}不一致' 
                if e.trigger:
                    indices = e.trigger.indices
                    trigger_chars = [s for s in e.trigger.text]
                    chars = [text[idx] for idx in indices]
                    assert trigger_chars == chars, f'事件触发词{trigger_chars}与标注触发词{chars}不一致'
        return v
    
    @validate_arguments
    def add_ent(self, ent: Entity):
        if not self.ents:
            self.ents = [ent]
        else:
            if ent not in self.ents:
                self.ents.append(ent)
            
    @validate_arguments
    def add_rel(self, rel: Relation):
        if not self.rels:
            self.rels = [rel]
        else:
            if rel not in self.rels:
                self.rels.append(rel)

    @validate_arguments
    def add_event(self, event: Event):
        if not self.events:
            self.events = [event]
        else:
            if event not in self.events:
                self.events.append(event)
            
    @validate_arguments
    def add_label(self, label: Label):
        if not self.labels:
            self.labels = [label]
        else:
            if label not in self.labels:
                self.labels.append(label)
    
    @validate_arguments
    def set_label(self, label: Label):
        self.label = label
        
    @validate_arguments
    def set_summary(self, summary: constr(strip_whitespace=True, strict=True, max_length=1)):
        self.summary = summary
        
    @validate_arguments
    def set_title(self, title: constr(strip_whitespace=True, strict=True, max_length=1)):
        self.title = title
        
    class Config:
        extra = 'forbid'
        allow_mutation = True
        
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other: "Doc") -> bool:
        if self.id and other.id:
            return self.id == other.id
        else:
            return self.label == other.label and \
                   self.labels == other.labels and \
                   self.ents == other.ents and \
                   self.rels == other.rels and \
                   self.events == other.events and \
                   self.summary == other.summary
                
                   
class DocBin():
    """存放所有doc
    参数:
    - docs (Union[List[Doc], Path]): 所有Doc文档示例或者jsonl格式地址
    """
    def __init__(self, docs : Union[List[Doc], Path] = None) -> None:
        super().__init__()
        if type(docs) == str:
            self._docs = self._get_docs_from_jsonl(docs)
        elif docs is None:
            self._docs = []
        else:
            self._docs = docs
        
    def save_jsonl(self, file_path: Path):
        """将数据以jsonl的格式保存到硬盘
        参数:
        - file_path (Path): 数据保存地址,例如./test.jsonl
        """
        path = Path(file_path)
        srsly.write_jsonl(path=path, lines=[doc.dict() for doc in self.docs])
    
    def _get_docs_from_jsonl(self, file_path: Path) -> List[Doc]:
        path = Path(file_path)
        datas = srsly.read_jsonl(path=path)
        docs = []     
        for d in datas:
            docs.append(Doc(**d))
        return docs
        
    def to_dataset(self, include: Optional[List] = None) -> Dataset:
        """转换数据集,None的数据自动去除
        参数:
        - include (List): 包含的字段名称,默认None
        """
        if include:
            datas = [doc.dict(include=set(include), exclude_none=True) for doc in self._docs]
        else:
            datas = [doc.dict(exclude_none=True) for doc in self._docs]
        return Dataset.from_pandas(pd.DataFrame.from_dict(datas))
    
    def to_dataframe(self, include: Optional[List] = None, dropna: bool = False) -> pd.DataFrame:
        """转换为dataframe格式
        """
        data = [doc.dict() for doc in self._docs]
        df = pd.DataFrame.from_records(data)
        if include:
            df = df.loc[:, include]
            if dropna:
                return df.dropna(how='all', axis=1)
            else:
                return df
        else:
            if dropna:
                return df.dropna(how='all', axis=1)
            else:
                return df
    
    def to_ner_dataset(self) -> Dataset:
        """转换为实体抽取数据集,自动过滤实体为None的doc
        """
        df = self.to_dataframe(include=['text', 'ents'])
        df = df[df['ents'].notna()]
        assert len(df)>0, '数据集为空'
        return Dataset.from_pandas(df, preserve_index=False)
    
    def to_tc_dataset(self) -> Dataset:
        """转换为单目标文本分类数据集
        """
        df = self.to_dataframe(include=['text', 'label'])
        df = df[df['label'].notna()]
        assert len(df)>0, '数据集为空'
        return Dataset.from_pandas(df, preserve_index=False)
    
    def to_re_dataset(self) -> Dataset:
        """转换为实体关系抽取数据集
        """
        df = self.to_dataframe(include=['text', 'rels'])
        print(df)
        df = df[df['rels'].notna()]
        assert len(df)>0, '数据集为空'
        return Dataset.from_pandas(df, preserve_index=False)
    
    def to_ee_dataset(self) -> Dataset:
        """转换为事件抽取数据集
        """
        df = self.to_dataframe(include=['text', 'events'])
        df = df[df['events'].notna()]
        assert len(df)>0, '数据集为空'
        return Dataset.from_pandas(df, preserve_index=False)
    
    def to_summary_dataset(self) -> Dataset:
        """转换为文本摘要数据集
        """
        df = self.to_dataframe(include=['text', 'summary'])
        df = df[df['summary'].notna()]
        assert len(df)>0, '数据集为空'
        return Dataset.from_pandas(df, preserve_index=False)
    
    def append(self, doc: Doc):
        self._docs.append(doc)
    
    def __getitem__(self, i):
        return self._docs[i]
    
    def __len__(self):
        return len(self._docs)

    def __repr__(self) -> str:
        return f"{len(self)} docs"