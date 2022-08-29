from ...utils.make_model import PLMBaseModel, align_token_span
from ...layers.dropout import MultiDropout
from ...metrics.triple import TripleF1, Triple
import torch.nn as nn
import torch


class OneRelClassifier(nn.Module):
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tag_size: int,
                 dropout: float):
        """用一个矩阵表示两两token关系

        Args:
            input_size (int): 一般为bert的hidden_size
            output_size (int): 关系类别
            tag_size (int): 矩阵每个位置的分类类别
            dropout(float): dropout的比率
        """
        super().__init__()
        self.output_size = output_size
        self.tag_size =tag_size
        self.lin1 = nn.Linear(input_size*2, input_size*2)
        self.lin2 = nn.Linear(input_size*2, tag_size*output_size)
        self.activation = nn.ReLU()  
        self.dropout = MultiDropout()     
    def forward(self, x): 
        batch_size, seq_len, hidden_size = x.size()
        # head: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,1,1, 2,2,2, 3,3,3
        head = x.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_size).reshape(batch_size, seq_len*seq_len, hidden_size)
        # tail: [batch_size, seq_len * seq_len, hidden_size] 重复样式为1,2,3, 1,2,3, 1,2,3
        tail = x.repeat(1, seq_len, 1)
        # 两两token的特征拼接在一起
        pairs = torch.cat([head, tail], dim=-1)
        pairs = self.lin1(pairs)
        pairs = self.dropout(pairs)
        pairs = self.activation(pairs)
        scores = self.lin2(pairs).reshape(batch_size, seq_len, seq_len,  self.output_size, self.tag_size)
        return scores



class OneRelForRelationExtraction(PLMBaseModel):
    def __init__(self,
                 lr: float,
                 scheduler: str,
                 dropout: float,
                 weight_decay: float,
                 **kwargs):
        super().__init__()
        
        self.plm = self.get_plm_architecture()
        self.dropout = MultiDropout()
        self.classifier = OneRelClassifier(input_size=self.plm.config.hidden_size, 
                                           output_size=len(self.hparams.label2id), 
                                           tag_size=len(self.hparams.tag2id),
                                           dropout=self.hparams.dropout)

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.train_metric = TripleF1()
        self.val_metric = TripleF1()
        self.test_metric = TripleF1()

        
    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.plm(input_ids=input_ids, 
                     token_type_ids=token_type_ids, 
                     attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def step(self, batch):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        target = batch['tag_ids']
        logits = self(input_ids, token_type_ids, attention_mask)
        loss = self.criterion(logits.permute(0, 4, 3, 1, 2), target)
        loss_mask = batch['loss_mask']
        loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask) 
        tag_ids = logits.argmax(-1).permute(0, 3, 1, 2)
        return loss, tag_ids

    def training_step(self, batch, batch_idx):
        loss, tag_ids = self.step(batch)
        self.log('train/loss', loss)
        return {'loss':loss}
    
        
    def validation_step(self, batch, batch_idx):
        loss, tag_ids = self.step(batch)
        batch_triples_x = self.extract_triples(tag_ids)
        batch_triples_y = self.extract_triples(batch['tag_ids'])
        self.val_metric(batch_triples_x, batch_triples_y)
        self.log('val/f1', self.val_metric, on_epoch=True, on_step=False, prog_bar=True)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr*50, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr*50, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler_config = self.get_scheduler_config(optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler_config]

    def extract_triples(self, tag_ids):
        # 原作者解码实现
        batch_tag_ids = torch.chunk(tag_ids, tag_ids.shape[0])
        batch_triples = []
        tag2id = self.hparams.tag2id
        id2rel = self.hparams.id2label
        for i in range(len(batch_tag_ids)):
            tag_ids = batch_tag_ids[i].squeeze(0)
            rel_numbers, seq_lens, seq_lens = tag_ids.shape
            relations, heads, tails = torch.where(tag_ids > 0)
            triples = set()
            pair_numbers = len(relations)
            if pair_numbers > 0:
                for i in range(pair_numbers):
                    r_index = relations[i]
                    h_start_index = heads[i]
                    t_start_index = tails[i]
                    # 如果当前第一个标签为HB-TB
                    if tag_ids[r_index][h_start_index][t_start_index] == tag2id[
                        'HB-TB'] and i + 1 < pair_numbers:
                        # 如果下一个标签为HB-TE
                        t_end_index = tails[i + 1]
                        if tag_ids[r_index][h_start_index][t_end_index] == tag2id['HB-TE']:
                            # 那么就向下找
                            for h_end_index in range(h_start_index, seq_lens):
                                # 向下找到了结尾位置
                                if tag_ids[r_index][h_end_index][t_end_index] == tag2id['HE-TE']:

                                    sub_head, sub_tail = h_start_index.item(), h_end_index
                                    obj_head, obj_tail = t_start_index.item(), t_end_index.item()
                                    rel = id2rel[r_index.item()]
                                    triples.add(Triple(triple=(sub_head, sub_tail+1, rel, obj_head, obj_tail+1)))
                                    break
            batch_triples.append(triples)
        return batch_triples
        # 改进版本解码
        # for i in range(len(bath_tag_ids)):
        #     triples = set()
        #     if len(torch.nonzero(bath_tag_ids[i])) >0:
        #         logits = bath_tag_ids[i].squeeze(0)
        #         rel_num, seq_len, seq_len = logits.shape
        #         rel2tag = {}
        #         for rel_id, start, end in torch.nonzero(logits):
        #             rel_id = rel_id.item()
        #             start = start.item()
        #             end = end.item()
        #             tag = self.hparams.id2tag[logits[rel_id, start, end].item()]
        #             if rel_id not in rel2tag:
        #                 rel2tag[rel_id] = [] 
        #             rel2tag[rel_id].append((tag, start, end))
        #         for rel_id, tags in rel2tag.items():
        #             rel = self.hparams.id2label[rel_id]
        #             for i, tag in enumerate(tags):
        #                 if tag[0] == 'HB-TB': 
        #                     if i == len(tags)-1:  # 主体客体都是一个token的情况
        #                         triples.add(Triple((tag[1], tag[1]+1, rel, tag[2], tag[2]+1)))
        #                     if i<len(tags)-1 :
        #                         if tags[i+1][0] == 'HB-TB':  # 此时客体是一个token, 继续搜索
        #                             for j in range(tag[1], seq_len):
        #                                 tag_id = logits[rel_id, j, tag[2]]
        #                                 tag_label = self.hparams.id2tag[tag_id.item()]
        #                                 if tag_label == 'HE-TE': # 客体是一个token
        #                                     triples.add(Triple((tag[1], j+1, rel, tag[2], tag[2]+1)))
        #                                     break
        #                                 if tag_label == 'HB-TB': # 说明主体客体都是一个token
        #                                     triples.add(Triple((tag[1], tag[1]+1, rel, tag[2], tag[2]+1)))
        #                                     break
        #                                 if j == seq_len-1 and tag != 'HB-TB' and tag!= 'HE-TE':  #最后一个也没有搜到说明主体客体都是一个token
        #                                     triples.add(Triple((tag[1], tag[1]+1, rel, tag[2], tag[2]+1)))
        #                                     break
        #                         if tags[i+1][0] == 'HB-TE': # 此时客体不是一个token, 继续搜索主体
        #                             for j in range(tags[i+1][1], seq_len):
        #                                 tag_id = logits[rel_id, j, tags[i+1][2]]
        #                                 tag_label = self.hparams.id2tag[tag_id.item()]
        #                                 if tag_label == 'HE-TE': # 找到主体
        #                                     triples.add(Triple((tags[i+1][1], j+1, rel, tag[2], tags[i+1][2]+1)))
        #                                     break
        #                                 if tag_label == 'HB-TB': # 说明主体一个token
        #                                     triples.add(Triple((tag[1], tag[1]+1, rel, tag[2], tag[2]+1)))
        #                                     break
        #                                 if j == seq_len-1 and tag != 'HB-TB' and tag!= 'HE-TE':  #最后一个也没有搜到说明主体是一个token
        #                                     triples.add(Triple((tag[1], tag[1]+1, rel, tag[2], tag[2]+1)))
        #                                     break
        #                         if tags[i+1][0] == 'HE-TE' and tag[1] == tags[i+1][1]: # 此时主体是一个token
        #                             triples.add(Triple((tag[1], tag[1]+1, rel, tag[2], tags[i+1][2])))
        #     batch_triples.append(triples)  
        # return batch_triples
                                                           
    def predict(self, text, device: str):
        max_length = min(len(text), self.hparams.max_length)
        inputs = self.tokenizer(
                text, 
                padding='max_length',  
                max_length=max_length,
                return_tensors='pt',
                truncation=True)
        inputs.to(torch.device(device))
        logits = self(**inputs)
        tag_ids = logits.argmax(-1).permute(0, 3, 1, 2)
        triples = self.extract_triples(tag_ids)
        rels = []
        for batch_triples in triples:
            if len(batch_triples)>0:
                offset_mapping = self.tokenizer(text,
                                                max_length=max_length,
                                                padding='max_length',
                                                truncation=True,
                                                return_offsets_mapping=True)['offset_mapping']
                for triple in batch_triples:
                    sub = align_token_span((triple[0], triple[1]), offset_mapping)
                    obj = align_token_span((triple[3], triple[4]), offset_mapping)
                    rels.append((sub[0],sub[1],triple[2],obj[0],obj[1]))
        return rels

        