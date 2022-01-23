import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
from ..layers import MultiLabelCategoricalCrossEntropy
from ...metrics.span import SpanEvaluator
from torchmetrics import MaxMetric
from torch.utils.data import DataLoader
from ...utils.preprocessing import fine_grade_tokenize

class BERTGlobalPointer(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        weight_decay: float,
        dropout: float ,
        **data_params
    ) : 
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.tokenizer = BertTokenizer.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.label2id = data_params['label2id']
        self.id2label = {v:k for k,v in self.label2id.items()}
        self.hidden_size = hidden_size
        self.fc = nn.Linear(self.bert.config.hidden_size, len(self.label2id) * hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.criterion = MultiLabelCategoricalCrossEntropy()
        self.metric = SpanEvaluator()
        self.train_best_f1 = MaxMetric()
        self.val_best_f1 = MaxMetric()

    
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings


    def forward(self, inputs):
        outputs = self.bert(**inputs).last_hidden_state
        batch_size = outputs.size()[0]
        seq_len = outputs.size()[1]
        outputs = self.dropout(self.fc(outputs))
        outputs = torch.split(outputs, self.hidden_size * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_size*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_size)
        qw, kw = outputs[...,:self.hidden_size], outputs[...,self.hidden_size:]

        # 添加位置信息
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.hidden_size)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        attention_mask = inputs['attention_mask']
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, len(self.label2id), seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1) 
        logits = logits - mask * 1e12
        return logits/self.hidden_size**0.5


    def shared_step(self, batch):
        inputs, span_ids = batch['inputs'], batch['span_ids']
        logits = self(inputs)
        batch_size, ent_type_size = logits.shape[:2]
        y_true = span_ids.reshape(batch_size*ent_type_size, -1)
        y_pred = logits.reshape(batch_size*ent_type_size, -1)
        loss = self.criterion(y_pred, y_true)
        return logits, loss

    
    def training_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        f1 = self.metric.get_sample_f1(logits, batch['span_ids'])
        # self.print(logits.shape)
        return {'loss': loss, 'batch_f1': f1}

    def training_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.train_best_f1.update(avg_f1)
        self.log('train/f1', avg_f1, prog_bar=True)
        self.log('train/best_f1', self.train_best_f1.compute(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        pred, loss = self.shared_step(batch)
        f1 = self.metric.get_sample_f1(pred, batch['span_ids'])
        return {'loss': loss, 'batch_f1': f1}

    def validation_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.val_best_f1.update(avg_f1)
        self.log('val/f1', avg_f1, prog_bar=True)
        self.log('val/best_f1', self.val_best_f1.compute(), prog_bar=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]


    def predict(self, text):
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens,
            is_pretokenized=True,
            add_special_tokens=True,
            return_tensors='pt')
        logits = self(inputs)
        spans_ls = torch.nonzero(logits>0).tolist()
        spans = []
        for span in spans_ls :
            start = span[2]
            end = span[3]
            spans.append([start-1, end-1, self.id2label[span[1]], text[start-1:end]])
        return spans



        

    


        


    

    
        
