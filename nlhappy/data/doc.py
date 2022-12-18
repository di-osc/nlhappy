from pydantic import BaseModel, conint, conint, constr, validator, conlist, conset, validate_arguments
from typing import List, Optional, Union, Tuple, DefaultDict, Dict, Any, Set, Generator
import srsly
from pathlib import Path
import pandas as pd
from .dataset import Dataset
from ..utils.text import split_sentence
from tqdm import tqdm
from functools import reduce
from operator import add


Label = constr(strip_whitespace=True, min_length=1)
Index = conint(ge=0, strict=True)


def assert_span_text_in_doc(doc_text: str, span_text: str, span_indices: List[Index]) -> None:
    """检查span的文本与标注的下标对应doc文本一致

    Args:
        doc_text (str): doc文本
        span_text (str): span文本
        span_indices (List[Index]): 在文档中的下标
    """
    try:
        text = ''.join([doc_text[i] for i in span_indices])
    except Exception as e:
        print(span_indices)
        print(len(span_indices))
        raise e
    # 如果实体文本存在则确保实体文本与下标对应文本一致,确保标注正确
    assert text == span_text, f'文本: <{span_text}> 与下标文本: <{text}> 不一致'
    
    
def strip_span(span_text: str, span_indices: List[Index]) -> Tuple[str, List[Index]]:
    """去除文本左右空格字符并将下标对齐

    Args:
        span_text (str): 文本
        span_indices (List[Index]): 下标

    Returns:
        Tuple[str, List[Index]]: 清洗后文本, 对齐后下标
    """
    ori_text = span_text
    text = span_text.strip()
    start = ori_text.index(text)
    indices = span_indices[start: start + len(text)]
    return text, indices
        
    
class Span(BaseModel):
    """原始文本的一个片段,可以连续也可以非连续
    
    参数:
    - text (str): 文本,默认None
    - indices (List[int]): 对应文本的下标
    
    说明:
    - 初始化时如果文本不为空,则会自动去除首尾的空格,
    - 下标会自动按照升序排列
    - 当文本去除首尾空格后,下标会自动修正,例如当text=' 中国'变为'中国', 下标[0,1,2]会变为[1,2]
    """
    text: constr(min_length=1)
    indices: Set[int]
    
    @property
    def is_continuous(self) -> bool:
        diff = []
        for i in range(1, len(self.indices)):
            diff.append((self.indices[i] - self.indices[i-1]) != 1)
        return not sum(diff) > 0
            
    @validator('text')
    def validate_text(cls, v: str, values: dict):
        if v is not None:
            values['ori_text'] = v # 记录原始文本,以此修改下边列表
            return v.strip() # 左右没有空格并且有效字符不为0则返回原文
        else:
            return v
    
    @validator('indices')
    def validate_indices(cls, v: List[Index], values):
        v = sorted(v)
        if 'text' in values:
            if values['text']:
                assert len(values['ori_text']) == len(v), f'下标: <{v}>与原始文本: <{values["ori_text"]}>长度不符'
                start = values['ori_text'].index(values['text'])
                indices = v[start: start+len(values['text'])]
                del values['ori_text']
                return indices
            else:
                return v
        else:
            return v
    
    def __hash__(self):
        return hash(self.text)
    
    def __gt__(self, other: "Span") -> bool:
        return self.indices[-1] > other.indices[0] 
    
    def __lt__(self, other: "Span") -> bool:
        return self.indices[-1] < other.indices[0]
    
    def __len__(self):
        return len(self.indices)
    
    def __add__(self, other: "Span"):
        if self.text is not None and other.text is not None:
            text = self.text + other.text
        else:
            text = None
        indices = self.indices + other.indices
        return Span(text=text, indices=indices)
    
    def __and__(self, other: "Span") -> "Span":
        indices = set(self.indices) & set(other.indices)
        if len(indices) == 0:
            return None
        else:
            indices = sorted(list(indices))
            chars = []
            for idx in indices:
                chars.append(self.text[self.indices.index(idx)])
            text = ''.join(chars)
            return Span(indices=indices, text=text)
        
    def __eq__(self, other: "Span") -> bool:
        if self.is_continuous and other.is_continuous:
            return self.indices[0] == other.indices[0] and self.indices[-1] == other.indices[-1]
        else:
            return self.indices == other.indices
    
    def __contains__(self, item: "Span") -> bool:
        if self.is_continuous and item.is_continuous:
            return self.indices[0] <= item.indices[0] and self.indices[-1] >= item.indices[-1]
        else:
            return sorted(self.indices)[0] <= sorted(item.indices)[0] and sorted(self.indices)[-1] >= sorted(item.indices)[-1]

        
class Entity(Span):
    """文本中有标签的Span,支持连续非连续类型
    参数:
    - text (str): 实体文本
    - label (str): 实体标签
    - indices (List[int]): 字符级别下标
    """
    label: Label = None
        
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
    
    args: conlist(item_type=Entity, min_items=1)
    label: Label
    trigger: Optional[Span] = None
    
    @validator('args')
    def validate_args(cls, v):
        return list(set(v))
    
    @validate_arguments
    def add_arg(self, arg: Entity):
        if arg not in self.args:
            self.args.append(arg)
    
    def __hash__(self):
        return hash(self.label)
    
    def __eq__(self, other: "Event") -> bool:
        return self.args == other.args and self.label == other.label and self.trigger == other.trigger
    
    
class Answer(BaseModel):
    spans: List[Span] = []
    
    @property
    def text(self):
        if len(self.spans) == 0:
            return ''
        else:
            text_list = []
            for span in self.spans:
                if span.text not in text_list and span.text is not None:
                    text_list.append(span.text)
            return ' '.join(text_list)
        
    def add_span(self, span: Span):
        if span not in self.spans:
            self.spans.append(span)
        self.spans = sorted(self.spans)


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
    - questions(Dict[str, Answer]): 问题对应答案
    """
    
    text: constr(min_length=1)
    id: Optional[str] = None
    label: Optional[Label] = None
    labels: Optional[List[Label]] = None
    ents: Optional[List[Entity]] = None
    rels: Optional[List[Relation]] = None
    events: Optional[List[Event]] = None   
    summary: Optional[str] = None
    title: Optional[str] = None
    questions : Optional[Dict[str, Answer]] = None
    
    @property
    def sents(self) -> Generator:
        start = 0
        for s in split_sentence(self.text, best=False):
            end = start + len(s)
            yield Span(text=s, indices=[i for i in range(start, end)])
            start = end
    
    @validator('text')
    def validate_text(cls, v: str):
        assert len(v.strip()) > 0, f'文本为空格'
        return v
    
    @validator('questions')
    def validate_questions(cls, v: Dict, values):
        for q, a in v.items():
            a: Answer
            for span in a.spans:
                assert type(span) == Span, '答案必须为Span'
                if span.text:
                    assert_span_text_in_doc(doc_text=values['text'], span_indices=span.indices, span_text=span.text)
        return v
    
    @validator('ents', each_item=True)
    def validate_ents(cls, v: Entity, values):      
        if v.text:
            assert_span_text_in_doc(doc_text=values['text'], span_indices=v.indices, span_text=v.text)          
        return v
    
    @validator('rels', each_item=True)
    def validate_rels(cls, v: Relation, values):
        text = values['text']
        if v.s.text:
            assert_span_text_in_doc(doc_text=text, span_text=v.s.text, span_indices=v.s.indices)
        if v.o.text:
            assert_span_text_in_doc(doc_text=text, span_text=v.o.text, span_indices=v.o.indices)
        return v
    
    @validator('events', each_item=True)
    def validate_events(cls, v: Event, values):
        text = values['text']
        for arg in v.args:
            arg: Entity
            if arg.text:
                assert_span_text_in_doc(doc_text=text, span_text=arg.text, span_indices=arg.indices) 
        if v.trigger:
            trigger : Span = v.trigger
            if trigger.text:
                assert_span_text_in_doc(doc_text=text, span_indices=trigger.indices, span_text=trigger.text)
        return v
    
    def _get_indices_text(self, indices: List[Index]) -> str:
        return ''.join([self.text[i] for i in indices])
    
    @validate_arguments
    def add_ent(self, ent: Entity):
        if ent.text:
            assert_span_text_in_doc(doc_text=self.text, span_indices=ent.indices, span_text=ent.text)
        else:
            ent_text = self._get_indices_text(indices=ent.indices)
            ent_text, ent_indices = strip_span(span_text=ent_text, span_indices=ent.indices)
            ent.text = ent_text
            ent.indices = ent_indices
        if not self.ents:
            self.ents = [ent]
        else:
            if ent not in self.ents:
                self.ents.append(ent)
            
    @validate_arguments
    def add_rel(self, rel: Relation):
        if rel.s.text:
            assert_span_text_in_doc(doc_text=self.text, span_indices=rel.s.indices, span_text=rel.s.text)
        else:
            s_text = self._get_indices_text(indices=rel.s.indices)
            s_text, s_indices = strip_span(span_text=s_text, span_indices=rel.s.indices)
            rel.s.text = s_text
            rel.s.indices = s_indices
        if rel.o.text:
            assert_span_text_in_doc(doc_text=self.text, span_indices=rel.o.indices, span_text=rel.o.text)
        else:
            o_text = self._get_indices_text(indices=rel.o.indices)
            o_text, o_indices = strip_span(span_text=o_text, span_indices=rel.o.indices)
            rel.o.text = o_text
            rel.o.indices = o_indices
        if not self.rels:
            self.rels = [rel]
        else:
            if rel not in self.rels:
                self.rels.append(rel)

    @validate_arguments
    def add_event(self, event: Event):
        for arg in event.args:
            arg: Entity
            if arg.text:
                assert_span_text_in_doc(doc_text=self.text, span_indices=arg.indices, span_text=arg.text)
            else:
                arg_text = self._get_indices_text(indices=arg.indices)
                arg_text, arg_indices = strip_span(span_text=arg_text, span_indices=arg.indices)
                arg.text = arg_text
                arg.indices = arg_indices
        if event.trigger:
            if event.trigger.text:
                assert_span_text_in_doc(doc_text=self.text, span_indices=event.trigger.indices, span_text=event.trigger.text)
            else:
                trigger_text = self._get_indices_text(indices=event.trigger.indices)
                trigger_text, trigger_indices = strip_span(span_text=trigger_text, span_indices=event.trigger.indices)
                event.trigger.text = trigger_text
                event.trigger.indices = trigger_indices
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
        
    @validate_arguments
    def add_question(self, question: constr(min_length=1, strip_whitespace=True)) -> None:
        if self.questions is None:
            self.questions = {}  
        if not question in self.questions:
            empty_ans = Answer()
            self.questions[question] = empty_ans
            
    def _combine_spans(self, spans: List[Span]) -> Optional[Span]:
        if len(spans) == 0:
            return None
        else:
            indices = [span.indices for span in spans]
            indices = reduce(add, indices)
            indices = sorted(list(set(indices)))
            text = self._get_indices_text(indices=indices)
            return Span(text=text, indices=indices)
        
    def combine_spans_to_contiguous(self, spans: List[Span]) -> Optional[Span]:
        if len(spans) == 0:
            return None
        else:
            start = spans[0].indices[0]
            end = spans[-1].indices[-1] + 1
            indices = [i for i in range(start, end)]
            text = self._get_indices_text(indices=indices)
            return Span(text=text, indices=indices)

    def split_by_sents(self, max_length: int) -> List[Span]:
        """将文本按照句子切分为不超过固定长度的片段

        Args:
            max_piece_length (int): 最大片段长度

        Returns:
            List[span]: 切分后的片段列表
        """
        pieces = []
        piece: List[Span] = []
        for sent in self.sents:
            sent: Span
            if len(piece)>0:
                cur_length = sent.indices[-1] + 1 - piece[0].indices[0]
            else:
                cur_length = sent.indices[-1] + 1
            if cur_length < max_length:
                piece.append(sent)
            else:
                if len(piece) > 0:
                    pieces.append(self.combine_spans_to_contiguous(spans=piece))
                if len(sent) > max_length:
                    continue
                else:
                    piece = [sent]
        if len(piece) > 0:
            pieces.append(self.combine_spans_to_contiguous(spans=piece))
        return pieces
    
    def get_answer(self, question: str) -> Answer:
        q = question.strip()
        return self.questions.get(q)
    
    def add_answer_span(self, question: str, answer_indices: List[int], answer_text: Optional[str] = None) -> None:
        question = question.strip()
        ans = self.questions.get(question)
        assert ans is not None, f'未能找到相应问题:{question},请确保问题已经设置'
        if answer_text is None:
            answer_text = self._get_indices_text(indices=answer_indices)
        else:
            assert_span_text_in_doc(doc_text=self.text, span_text=answer_text, span_indices=answer_indices)
        span = Span(indices=answer_indices, text=answer_text)
        ans.add_span(span=span)
        
        
    class Config:
        extra = 'forbid'
        allow_mutation = True
        
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other: "Doc") -> bool:
        if self.id and other.id:
            return self.id == other.id
        else:
            return self.text == other.text and \
                   self.label == other.label and \
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
        srsly.write_jsonl(path=path, lines=[doc.dict() for doc in self._docs])
    
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
    
    def to_qa_dataset(self, max_length: int = 450, only_have_answer: bool = False) -> Dataset:
        """转换为问答数据集

        Args:
            max_length (int, optional): 按照句子级别切分的最大文本长度. Defaults to 500.

        Returns:
            Dataset : 按照句子切分后的问答数据集
        """
        data = {'text': [], 'question': [], 'answer': [], 'spans': []}
        for doc in tqdm(self._docs):
            doc: Doc
            pieces = doc.split_by_sents(max_length=max_length)
            for p in pieces:
                p: Span
                for q, a in doc.questions.items():
                    a: Answer
                    new_a = Answer()
                    for span in a.spans:
                        _span = span & p
                        if _span:
                            indices = [i - p.indices[0] for i in _span.indices]
                            p_span = Span(text=_span.text, indices=indices)
                            assert_span_text_in_doc(doc_text=p.text, span_text=p_span.text, span_indices=p_span.indices)
                            new_a.add_span(span=p_span)
                    if only_have_answer:
                        if len(new_a.spans) > 0:
                            data['text'].append(p.text)
                            data['question'].append(q)
                            data['answer'].append(new_a.text)
                            data['spans'].append(new_a.dict()['spans'])  
                    else:
                        data['text'].append(p.text)
                        data['question'].append(q)
                        data['answer'].append(new_a.text)
                        data['spans'].append(new_a.dict()['spans'])
        return Dataset.from_dict(data)
    
    @validate_arguments
    def append(self, doc: Doc):
        self._docs.append(doc)
    
    @validate_arguments
    def add(self, doc: Doc):
        if doc not in self._docs:
            self._docs.append(doc)
    
    def __getitem__(self, i):
        return self._docs[i]
    
    def __len__(self):
        return len(self._docs)
    
    def __repr__(self) -> str:
        return f"{len(self._docs)} docs"
    
    def __str__(self) -> str:
        return f"{len(self._docs)} docs"
    
    def __add__(self, other: "DocBin") -> "DocBin":
        return DocBin(self._docs + other._docs)