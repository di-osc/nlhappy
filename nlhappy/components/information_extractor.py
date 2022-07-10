from spacy.tokens import Doc
import os
from typing import Dict, List
from spacy.lang.zh import Chinese
from spacy.language import Language
from ..utils.make_doc import EventData, RelationData
from ..utils.utils import get_logger
from spacy.util import filter_spans
from ..models.prompt_span_extraction import BERTGlobalSpan
import pickle

log = get_logger()


class PSEExtractor:
    def __init__(self, 
                 nlp: Language, 
                 name: str,  
                 device: str,
                 schemas: List[str], 
                 threshold: float =0.5, 
                 set_ents: bool =True,
                 num_sentences: int =0,
                 stride: int =-1):
        self.pipe_name = name
        self.set_ents = set_ents
        self.num_sentences = num_sentences
        self.stride = stride
        self.threshold = threshold
        self.schemas = schemas
        self.device = device

          
        
    def __call__(self, doc: Doc) -> Doc:
        if self.num_sentences > 0:
            raise NotImplementedError
        else:
            #schemas = {'entity':[],'relation':{},'event':{}}
            if 'entity' in self.schemas:
                ent_prompts = self.schemas['entity']
                texts = [doc.text for _ in range(len(ent_prompts))]
                spans = self.model.predict(ent_prompts, texts, device=self.device,threshold=self.threshold)
                if len(spans)>0:
                    doc.spans['all'] = []
                    for span in spans:
                        try:
                            doc.spans['all'].append(doc.char_span(span[0], span[1], label=span[2]))
                        except:
                            log.warning(f'found bad span ({span[0]}, {span[1]}). skip set entity')     
                        if self.set_ents:
                            doc.set_ents(filter_spans(doc.spans['all']))
            if 'relation' in self.schemas:
                sub_prompts = list(self.schemas['relation'].keys())
                texts = texts = [doc.text for _ in range(len(sub_prompts))]
                sub_spans = self.model.predict(sub_prompts, 
                                            texts, 
                                            device=self.device,
                                            threshold=self.threshold)
                if self.set_ents:
                    try:
                        ents = [doc.char_span(s[0], s[1], s[2]) for s in sub_spans]
                        doc.set_ents(filter_spans(ents))
                    except:
                        log.info(f'try set ents on doc({doc.text}) failed')
                for sub_span in sub_spans:
                    sub_label = sub_span[2]
                    sub_text = sub_span[3]
                    rel_types = self.schemas['relation'][sub_label]
                    rel_prompts = [sub_text + '的' + rel_type for rel_type in rel_types]
                    texts = [doc.text for _ in range(len(rel_types))]
                    obj_spans = self.model.predict(rel_prompts, 
                                            texts, 
                                            device=self.device,
                                            threshold=self.threshold)  
                    if len(obj_spans) >0:
                        label_mapping = {}
                        for obj in obj_spans:
                            rel_type = obj[2].split(sub_text)[-1][1:]
                            if rel_type not in label_mapping:
                                label_mapping[rel_type] = []
                            if obj[0]<=obj[1]:
                                label_mapping[rel_type].append((obj[0],obj[1]))
                        sub = (sub_span[0], sub_span[1])
                        for rel_type in label_mapping:
                            obj_offsets = label_mapping[rel_type]
                            doc._.rel_data.append(RelationData(sub, label=rel_type, objs=obj_offsets))
            if 'event' in self.schemas:
                events = list(self.schemas['event'].keys())
                trigger_prompts = [event + '的触发词' for event in events]
                texts = [doc.text for _ in range(len(trigger_prompts))]
                trigger_spans = self.model.predict(trigger_prompts, texts,device=self.device, threshold=self.threshold)
                if len(trigger_spans) > 0:
                    for trigger_span in trigger_spans:
                        event = trigger_span[2].split('的触发词')[0]
                        args = self.schemas['event'][event]
                        args_dict = {}
                        args_dict['触发词'] = [(trigger_span[0], trigger_span[1])]
                        arg_prompts = [event+'的'+arg for arg in args]
                        texts = [doc.text for _ in range(len(arg_prompts))]
                        arg_spans = self.model.predict(prompts=arg_prompts,
                                                       texts=texts,
                                                       device=self.device,
                                                       threshold=self.threshold)
                        for arg_span in arg_spans:
                            arg_type = arg_span[2].split(event+'的')[-1]
                            if arg_type not in args_dict:
                                args_dict[arg_type] = []
                            if arg_span[0] <= arg_span[1]:
                                args_dict[arg_type].append((arg_span[0], arg_span[1]))
                        if len(args_dict)>0:
                            doc._.event_data.append(EventData(label=event, roles=args_dict))         
        return doc
    

    def to_disk(self, path:str, exclude):
        # 复制原来模型参数到新的路径
        # path : save_path/information_extractor
        if not os.path.exists(path):
            os.mkdir(path=path)
        model = 'pse.pkl'
        model_path = os.path.join(path, model)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
    def from_disk(self, path:str, exclude):
        model_path = os.path.join(path, 'pse.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model.freeze()
        try:
            self.model.to(self.device)
        except:
            log.info(f' to device {self.device} failed')
        
    def init_model(self, model_or_path):
        if isinstance(model_or_path, BERTGlobalSpan):
            self.model = model_or_path
            self.model.freeze()
            self.model.to(self.device)
            
        else:
            self.model= BERTGlobalSpan.load_from_checkpoint(model_or_path)
            self.model.freeze()
            self.model.to(self.device)
        
default_config = {'threshold': 0.5, 
                  'set_ents': True, 
                  'num_sentences': 0,
                  'stride': -1,
                  'device':'cpu'}
        
        
@Chinese.factory('information_extractor',assigns=['doc.spans','doc.ents','doc._.relations','doc._.events'],default_config=default_config)
def make_ie(nlp: Language,
            name:str, 
            schemas: Dict,  
            num_sentences:bool,
            stride:int, 
            threshold:float, 
            set_ents: bool,
            device:str):
    """the information extractor pipe

    Args:
        nlp (Language): spacy nlp
        name (str): name of the pipe
        schemas (Dict): schemas of the pipe
        num_sentences (bool): num of sentences every time to extract
        threshold (float): threshold of the model
        set_ents (bool): whether to set the doc.ents

    Returns:
        PSEExtractor: extractor based on the prompt-span-extraction model
    """
    
    return PSEExtractor(nlp=nlp, 
                        name=name,
                        schemas=schemas, 
                        num_sentences=num_sentences,
                        stride=stride, 
                        threshold=threshold, 
                        set_ents=set_ents,
                        device=device)
   