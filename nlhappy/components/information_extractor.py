import shutil
from spacy.tokens import Doc
import os
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
from typing import Dict, List
from spacy.lang.zh import Chinese
from spacy.language import Language
import numpy as np
from ..utils.make_doc import Event, Relation
from ..utils.utils import get_logger
from spacy.util import filter_spans

log = get_logger()


class PSEONNXExtractor:
    def __init__(self, 
                 nlp: Language, 
                 name: str,  
                 ckpt: str,
                 schemas: List[str], 
                 threshold: float =0.5, 
                 set_ents: bool =True,
                 num_sentences: int =0,
                 stride: int =-1,
                 model:str = 'pse.onnx',
                 tokenizer:str = 'tokenizer'):
        self.pipe_name = name
        self.ckpt = ckpt
        self.set_ents = set_ents
        self.num_sentences = num_sentences
        self.stride = stride
        self.threshold = threshold
        self.model_name = model
        self.schemas = schemas
        self.tokenizer = tokenizer
        infer_path = os.path.join(self.ckpt, self.model_name)
        if not os.path.exists(infer_path):
            log.warning(f'cannot load infer model from {infer_path}')
        else:
            self.infer = InferenceSession(infer_path)
        tokenizer_path = os.path.join(self.ckpt, self.tokenizer)
        if not os.path.exists(tokenizer_path):
            log.warning(f"cannot load tokenizer from {tokenizer_path}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        
        
        
    def __call__(self, doc: Doc) -> Doc:
        if self.num_sentences > 0:
            raise NotImplementedError
        else:
            #schemas = {'entity':[],'relation':{},'event':{}}
            if 'entity' in self.schemas:
                ent_prompts = self.schemas['entity']
                texts = [doc.text for _ in range(len(ent_prompts))]
                label_idxs, starts, ends = self.predict(ent_prompts, texts)
                doc.spans['all'] = []
                if len(label_idxs) > 0:
                    for i in range(len(label_idxs)):
                        label = ent_prompts[label_idxs[i]]
                        start = starts[i]
                        end = ends[i]
                        try:
                            doc.spans['all'].append(doc.char_span(start, end, label=label))
                        except:
                            log.warning(f'found bad span ({start}, {end}). skip set entity')     
                    if self.set_ents:
                        doc.set_ents(filter_spans(doc.spans['all']))
            if 'relation' in self.schemas:
                for sub_type in self.schemas['relation']:
                    sub_prompt = [sub_type]
                    text = [doc.text]
                    label_idxs, starts, ends = self.predict(sub_prompt, text)
                    sub_spans = [doc.char_span(starts[i], ends[i], label=sub_type) for i in range(len(label_idxs))]
                    for span in sub_spans:
                        for rel_type in self.schemas['relation'][sub_type]:
                            prompts = [span.text + '的' + rel_type]
                            texts = [doc.text]
                            idxs, starts, ends = self.predict(prompts, texts)
                            obj_spans = [doc.char_span(starts[i], ends[i], label=prompts[0]) for i in range(len(idxs))]
                            if len(obj_spans) >0:
                                doc._.relations.append(Relation(rel_type,span, obj_spans))
            if 'event' in self.schemas:
                for event in self.schemas['event']:
                    args = self.schemas['event'][event]
                    args_dict = {}
                    for arg in args:
                        prompts = [event+'事件的'+arg]
                        texts = [doc.text]
                        idxs, starts, ends = self.predict(prompts, texts)
                        spans = [doc.char_span(starts[i], ends[i], label=prompts[0]) for i in range(len(idxs))]
                        args_dict[arg] = spans
                    doc._.events.append(Event(event, args_dict))            
        return doc
    
    def predict(self, prompts: List[str], texts: List[str]):
        inputs = self.tokenizer(prompts, texts, return_tensors='np', return_offsets_mapping=True, padding=True)
        offset_mapping = inputs['offset_mapping']
        del inputs['offset_mapping']
        logits = self.infer.run(None, dict(inputs))[0]
        res = np.nonzero(logits>self.threshold)
        idxs, _, starts, ends = res[0].tolist(), res[1].tolist(), res[2].tolist(), res[3].tolist()
        starts = [offset_mapping[idxs[i]][starts[i]][0] for i in range(len(starts))]
        ends = [offset_mapping[idxs[i]][ends[i]][1] for i in range(len(ends))]
        return idxs, starts, ends

    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        # path : save_path/information_extractor
        shutil.copytree(self.ckpt, path)
        
    def from_disk(self, path:str, exclude):
        # path: load_path/information_extractor
        infer_path = os.path.join(path, self.model_name)
        self.infer = InferenceSession(infer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        
default_config = {'threshold': 0.5, 
                  'set_ents': True, 
                  'num_sentences': 0,
                  'stride': -1, 
                  'model': 'pse.onnx',
                  'use_onnx': True}
        
        
@Chinese.factory('information_extractor',assigns=['doc.spans','doc.ents','doc._.relations','doc._.events'],default_config=default_config)
def make_ie(nlp: Language,
            name:str, 
            schemas: Dict, 
            model:str, 
            ckpt:str, 
            num_sentences:bool,
            stride:int, 
            threshold:float, 
            set_ents: bool,
            use_onnx: bool = True):
    """the information extractor pipe

    Args:
        nlp (Language): spacy nlp
        name (str): name of the pipe
        schemas (Dict): schemas of the pipe
        model (str): name of the model
        ckpt (str): path of the model
        num_sentences (bool): num of sentences every time to extract
        threshold (float): threshold of the model
        set_ents (bool): whether to set the doc.ents

    Returns:
        PSEExtractor: extractor based on the prompt-span-extraction model
    """
    if use_onnx:
        return PSEONNXExtractor(nlp=nlp, 
                                name=name,
                                schemas=schemas,
                                model=model, 
                                ckpt=ckpt, 
                                num_sentences=num_sentences,
                                stride=stride, 
                                threshold=threshold, 
                                set_ents=set_ents)
    else:
        raise NotImplementedError