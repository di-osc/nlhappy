import pytorch_lightning as pl
from ..layers import CRF
import torch
from ...metrics.chunk import f1_score, get_entities
from transformers import BertModel, BertTokenizer
from torchmetrics import MaxMetric
import torch.nn as nn
from ...utils.preprocessing import fine_grade_tokenize

class BertCRF(pl.LightningModule):
    def __init__(self,
                 hidden_size: int,
                 lr: float,
                 crf_lr: float,
                 weight_decay: float,
                 dropout: float,
                 **data_params):
        super().__init__()
        self.save_hyperparameters()
        self.label2id = data_params['label2id']
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.lr = lr
        self.crf_lr = crf_lr
        self.weight_decay = weight_decay
        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.classifier = torch.nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.LayerNorm(normalized_shape=hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(self.label2id))
        )
        self.crf = CRF(len(self.label2id))
        self.tokenizer = BertTokenizer.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.metric = f1_score
        self.val_f1_best = MaxMetric()
        self.train_f1_best = MaxMetric()
        

    def forward(self, inputs, label_ids=None):
        x = self.bert(**inputs).last_hidden_state
        emissions = self.classifier(x)
        mask = inputs['attention_mask'].gt(0)
        if label_ids is not None :
            loss = self.crf(emissions, label_ids, mask=mask)
            pred_ids= self.crf.decode(emissions, mask=mask)
            return loss, pred_ids
        else :
            return self.crf.decode(emissions, mask=mask)
        

    
    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        label_ids = batch['label_ids']
        mask = label_ids.gt(-1)
        #这里label_ids 用-100 进行pad 改为 0
        label_ids[label_ids == -100] = -1
        log_like, pred_ids= self(inputs, label_ids, mask=mask)
        loss = log_like * (-1)
        
        pred_labels = []
        for ids in pred_ids:
            pred_labels.append([self.id2label[id] for id in ids])

        labels = []
        for i in range(len(label_ids)):
            indice = torch.where(label_ids[i] >= 0)
            ids = label_ids[i][indice].tolist()
            labels.append([self.id2label[id] for id in ids])
        f1 = self.metric(y_true=labels, y_pred=pred_labels)
        return {'loss':loss, 'batch_f1': f1}

    def training_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.train_f1_best.update(avg_f1)
        self.log('train/f1', avg_f1, prog_bar=True)
        self.log('train/best_f1', self.train_f1_best.compute(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        label_ids = batch['label_ids']
        mask = label_ids.gt(-1)
        #这里label_ids 用-100 进行pad 改为 -1
        label_ids[label_ids == -100] = -1
        log_like, pred_ids= self(inputs, label_ids, mask=mask) 
        loss = log_like * (-1)
        
        pred_labels = []
        for ids in pred_ids:
            pred_labels.append([self.id2label[id] for id in ids])

        labels = []
        for i in range(len(label_ids)):
            indice = torch.where(label_ids[i] >= 0)
            ids = label_ids[i][indice].tolist()
            labels.append([self.id2label[id] for id in ids])
        f1 = self.metric(y_true=labels, y_pred=pred_labels)
        return {'loss':loss, 'batch_f1': f1}

    def validation_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.val_f1_best.update(avg_f1)
        self.log('val/f1', avg_f1, prog_bar=True)
        self.log('val/best_f1', self.val_f1_best.compute(), prog_bar=True)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.lr * 5, 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.lr * 5, 'weight_decay': 0.0},
            {'params': self.crf.parameters(), 'lr': self.crf_lr}
        ]
        self.optimizer = torch.optim.Adam(grouped_parameters, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]


    def predict(self, text: str):
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens,
            is_pretokenized=True,
            add_special_tokens=True,
            return_tensors='pt')
        outputs = self(inputs)
        labels = [self.id2label[id] for id in outputs[0]]
        ents = get_entities(seq=labels[1:-1]) 
        new_ents = []
        for ent in ents:
            new_ents.append([ent[0], ent[1], ent[2], text[ent[1]:ent[2]+1]])
        return new_ents
        
