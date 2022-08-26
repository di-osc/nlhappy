from re import T
from typing import List
from ...utils.make_model import PLMBaseModel
from ...layers.dropout import MultiDropout
from ...metrics.triple import TripleF1, Triple
from ...metrics.span import SpanOffsetF1
import torch.nn as nn
import torch
import torch.nn.functional as F


class CasRelLoss(nn.BCELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    def forward(self, subs_x, subs_y, objs_x, objs_y, mask):

        # sujuect部分loss
        subject_loss = super().forward(subs_x, subs_y)
        subject_loss = subject_loss.mean(dim=-1)
        subject_loss = (subject_loss * mask).sum() / mask.sum()
        # object部分loss
        object_loss = super().forward(objs_x, objs_y)
        object_loss = object_loss.mean(dim=-1).sum(dim=-1)
        object_loss = (object_loss * mask).sum() / mask.sum()
        return subject_loss + object_loss



class CasRelForRelationExtraction(PLMBaseModel):
    """CasRel 关系抽取模型, 无法解决嵌套实体的情况

    Args:
        - lr: 学习率
        - dropout: dropout的比率
        - scheduler: 学习率衰减策略必须
        - threshold: 阈值
    Reference:
        [1] https://github.com/xiangking/ark-nlp/blob/main/ark_nlp/model/re/prgc_bert/prgc_bert.py
    """
    
    def __init__(self,
                 lr: float,
                 dropout: float = 0.2,
                 weight_decay: float = 0.01,
                 scheduler: str = 'linear_warmup_step',
                 threshold: float = 0.5,
                 **kwargs):
        super().__init__()

        self.plm = self.get_plm_architecture()
        
        self.dropout = MultiDropout()
        # self.condLayerNorm = LayerNorm(hidden_size=self.plm.config.hidden_size, conditional_size=self.plm.config.hidden_size*2)
        self.sub_classifier = nn.Linear(self.plm.config.hidden_size, 2)
        self.obj_classifier = nn.Linear(self.plm.config.hidden_size, len(self.hparams.label2id)*2)

        self.sub_criterion = nn.BCEWithLogitsLoss()
        self.obj_criterion = nn.BCEWithLogitsLoss()
        
        self.triple_metric = TripleF1()
        self.sub_metric = SpanOffsetF1()
        

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                **kwargs):
        batch_size, seq_len = input_ids.shape
        batch_lengths = torch.sum(attention_mask, -1)
        batch_hidden_state = self.encode_step(input_ids, token_type_ids, attention_mask)
        batch_sub_logits = self.sub_classifier_step(batch_hidden_state)
        batch_sub_scores = torch.sigmoid(batch_sub_logits)
        batch_sub_spans = self.get_sub_spans(batch_sub_scores, attention_mask, self.hparams.threshold)
        batch_relations = []
        for i in range(batch_size):
            sub_spans = batch_sub_spans[i]
            length = batch_lengths[i]
            hidden_state = batch_hidden_state[i].unsqueeze(0).repeat(len(sub_spans), 1, 1) # 改变回原来形状batch, seq, hidden
            if sub_spans:
                # sub_spans : num_spans, 2
                sub_spans = torch.tensor(sub_spans, dtype=torch.long, device=self.device)
                obj_logits = self.obj_classifier_step(hidden_state, sub_spans)
                relations = self.get_triples(obj_logits, sub_spans, length)
                batch_relations.append(relations)
            else:
                batch_relations.append(set())
        return batch_relations, batch_sub_spans



        
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        subs = batch['subs']
        sub = batch['sub']
        objs = batch['objs']
        last_hidden_state = self.encode_step(input_ids, token_type_ids, attention_mask)
        sub_logits = self.sub_classifier_step(last_hidden_state)
        obj_logits = self.obj_classifier_step(last_hidden_state, sub)
        subject_loss = self.sub_criterion(sub_logits, subs)
        subject_loss = subject_loss.mean(dim=-1)
        subject_loss = (subject_loss * attention_mask).sum() / attention_mask.sum()
        object_loss = self.obj_criterion(obj_logits, objs)
        object_loss = object_loss.mean(dim=-1).sum(dim=-1)
        object_loss = (object_loss * attention_mask).sum() / attention_mask.sum()
        loss = subject_loss + object_loss
        self.log('train/loss', loss)
        return {'loss':loss}

        
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        batch_size, seq_len = input_ids.shape
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        batch_objs = batch['objs']
        batch_sub = batch['sub']
        batch_subs_true = self.get_sub_spans(batch['subs'], attention_mask, self.hparams.threshold)
        batch_relations, batch_subs = self(input_ids, token_type_ids, attention_mask)
        batch_triples = []
        for i in range(len(batch_relations)):
            triples = batch_relations[i]
            ts = set()
            for t in triples:
                ts.add(Triple(t))
            batch_triples.append(ts)
        batch_triples_true = []
        for i in range(batch_size):
            triples = self.get_triples(batch_objs[i].unsqueeze(0), batch_sub[i].unsqueeze(0), torch.sum(attention_mask[i]))
            batch_triples_true.append(set([Triple(t) for t in triples]))
                        
        self.triple_metric(batch_triples, batch_triples_true)
        self.sub_metric(batch_subs, batch_subs_true)
        self.log('triple/f1', self.triple_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('sub/f1', self.sub_metric, on_step=False, on_epoch=True, prog_bar=True)


        
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.sub_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.sub_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.obj_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.obj_classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': self.hparams.lr, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, eps=1e-5)
        scheduler_config = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler_config]
    
    def encode_step(self, input_ids, token_type_ids, attention_mask):
        last_hidden_state = self.plm(input_ids, token_type_ids, attention_mask).last_hidden_state
        return last_hidden_state
    
    
    def sub_classifier_step(self, last_hidden_state):
        # sub_logits = (torch.sigmoid(self.sub_classifier(last_hidden_state)))**2
        sub_logits = self.sub_classifier(last_hidden_state)
        # sub_scores = torch.sigmoid(sub_logits)
        return sub_logits

    
    def obj_classifier_step(self, last_hidden_state, sub_spans):
        sub_reps = self.get_sub_representation(last_hidden_state, sub_spans)
        # output = self.condLayerNorm([last_hidden_state, sub_reps])
        # output = (torch.sigmoid(self.obj_classifier(output)))**4
        output = last_hidden_state + sub_reps
        output =self.obj_classifier(output)
        obj_logits = output.reshape(*output.shape[:2], len(self.hparams.label2id), 2)
        return obj_logits
    
    
    @staticmethod
    def get_sub_representation(hidden_states, sub_spans):
        """根据sub从output中取出subject的向量表征
        hidden_state: batch, seq, hidden
        sub_spans: batch, num_subs, 2

        """
        start = torch.gather(hidden_states, dim=1, index=sub_spans[:, :1].unsqueeze(2).expand(-1, -1, hidden_states.shape[-1]))
        end = torch.gather(hidden_states, dim=1, index=sub_spans[:, 1:].unsqueeze(2).expand(-1, -1, hidden_states.shape[-1]))
        # subject = torch.cat([start, end], 2)
        sub_rep = (start + end) / 2
        
        return sub_rep
        

    def get_sub_spans(self, sub_scores, mask, threshold) -> List:
        """将sub classifier 线性层经过sigmoid输出的score转换为sub spans
        返回一个列表
        """
        lengths = torch.sum(mask, -1)
        starts = sub_scores[:, :, :1]
        ends = sub_scores[:, :, 1:]
        sub_spans = []
        for start, end, l in zip(starts, ends, lengths):
            tmp = set()
            start = start.squeeze()[:l]
            end = end.squeeze()[:l]
            for i, st in enumerate(start):
                if st > threshold:
                    s = i
                    for j in range(i, l):
                        if end[j] > threshold:
                            e = j
                            if (s,e) not in sub_spans:
                                tmp.add((s,e))
                            break

            sub_spans.append(tmp)
        return sub_spans

    def get_triples(self, obj_logits, sub_spans, length):
        """_summary_

        Args:
            obj_logits (_type_): num_subs, seq_len, num_rels, 2
            sub_spans (_type_): num_subs, 2
            length (_type_): _description_

        Returns:
            _type_: _description_
        """
        num_label = obj_logits.shape[2]
        num_subs = len(sub_spans)
        triples = set()
        subjects = set()
        # objects = []
        for b in range(num_subs):
            tmp = obj_logits[b, ...]
            start, end = sub_spans[b]
            subject = (start.item(), end.item())
            if subject not in subjects:
                subjects.add(subject)
                for label_id in range(num_label):
                    start = tmp[:, label_id, :1]
                    end = tmp[:, label_id, 1:]
                    start = start.squeeze()[:length]
                    end = end.squeeze()[:length]
                    for i, st in enumerate(start):
                        if st > self.hparams.threshold:
                            s = i
                            for j in range(i, length):
                                if end[j] > 0.5:
                                    e = j
                                    object = (s,e)
                                    # if object not in objects:
                                    #     objects.append(object)
                                    if (*subject, label_id, *object) not in triples:
                                        triples.add((*subject, label_id, *object))
                                    break
        # print(triples) 
        return triples