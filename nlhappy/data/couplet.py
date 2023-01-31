from pydantic import BaseModel, constr, validator, validate_arguments
from typing import List, Union, Optional
from .dataset import Dataset
import pandas as pd
from pathlib import Path
import srsly


class Couplet(BaseModel):
    """一副对联
    Args:
        left: 上联
        right: 下联
        top: 横批
    """
    left: constr(strip_whitespace=True, strict=True, min_length=1)
    right: constr(strip_whitespace=True, strict=True, min_length=1)
    top: constr(strip_whitespace=True, strict=True, min_length=0) = None
    
    @validator('right')
    def validate_right(cls, v: str, values) -> str:
        left = values['left']
        right = v.strip()
        assert len(right) == len(left), '上联与下联字数不同'
        return right

class CoupletBin:
    """保存对联
    """
    def __init__(self, couplets: Union[List[Couplet], Path] = []) -> None:
        super().__init__()
        if type(couplets) == str:
            self._couplets = self._get_docs_from_jsonl(couplets)
        elif couplets is None:
            self._couplets = []
        else:
            self._couplets = couplets
    
    def __getitem__(self, i):
        return self._couplets[i]
    
    def __len__(self):
        return len(self._couplets)
    
    def __repr__(self) -> str:
        return f"{len(self._couplets)} couplets"
    
    def __str__(self) -> str:
        return f"{len(self._couplets)} couplets"
    
    def __add__(self, other: Union["CoupletBin", List[Couplet]]) -> "CoupletBin":
        if isinstance(other, list):
            return CoupletBin(self._couplets + other)
        if isinstance(other, CoupletBin):
            return CoupletBin(self._couplets + other._couplets)
    
    @validate_arguments
    def append(self, couplet: Couplet):
        self._couplets.append(couplet)
        return self
        
    @validate_arguments
    def add(self, couplet: Couplet):
        if couplet not in self._couplets:
            self._couplets.append(couplet)
        return self
        
    def save_to_disk(self, file_path: Path):
        """将数据以jsonl的格式保存到硬盘
        参数:
        - file_path (Path): 数据保存地址,例如./test.jsonl
        """
        path = Path(file_path)
        srsly.write_jsonl(path=path, lines=[couplet.dict() for couplet in self._couplets])
    
    def _get_docs_from_jsonl(self, file_path: Path) -> List[Couplet]:
        path = Path(file_path)
        lines = srsly.read_jsonl(path=path)
        couplets = []     
        for l in lines:
            couplets.append(Couplet(**l))
        return couplets
        
    def to_dataset(self, include: Optional[List] = None) -> Dataset:
        """转换数据集,None的数据自动去除
        参数:
        - include (List): 包含的字段名称,默认None
        """
        if include:
            data = [c.dict(include=set(include), exclude_none=True) for c in self._couplets]
        else:
            data = [c.dict(exclude_none=True) for c in self._couplets]
        return Dataset.from_pandas(pd.DataFrame.from_dict(data))