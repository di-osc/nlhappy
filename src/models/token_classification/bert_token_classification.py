import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from datasets import load_metric
import torch
import torch.nn.functional as F
import os
from ...metrics.chunk import f1_score
from torchmetrics import MaxMetric
from ...layers import SimpleDense


class BertTokenClassification(pl.LightningModule):
    def __init__(self,
                 hidden_size: int,
                 lr: float,
                 weight_decay: float,
                 dropout:float =0.5,
                 **data_params):
        super().__init__()
        self.save_hyperparameters()
        self.label2id = data_params['label2id']
        self.id2label = {id: label for label, id in self.label2id.items()}
        

        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(self.label2id))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = f1_score
        self.train_best_f1 = MaxMetric()
        self.val_best_f1 = MaxMetric()
        
        
    
    def forward(self, inputs):
        bert_hidden_state = self.bert(**inputs).last_hidden_state
        logits = self.classifier(bert_hidden_state)
        return logits

    def shared_step(self, batch):
        inputs = batch['inputs']
        label_ids = batch['label_ids']
        logits = self(inputs)
        mask = torch.gt(label_ids, -1)
        loss = self.criterion(logits.view(-1, len(self.label2id))[mask.view(-1)], label_ids.view(-1)[mask.view(-1)])
        pred_ids = torch.argmax(logits, dim=-1)
        labels = []
        pred_labels = []
        for i in range(len(label_ids)):
            indice = torch.where(label_ids[i] >= 0)
            ids = label_ids[i][indice].tolist()
            labels.append([self.id2label[id] for id in ids])
            pred = pred_ids[i][indice].tolist()
            pred_labels.append([self.id2label[id] for id in pred])
        return loss, labels, pred_labels



    def training_step(self, batch, batch_idx):
        loss, labels, pred_labels = self.shared_step(batch)
        f1 = self.metric(y_true=labels, y_pred=pred_labels)
        return {'loss':loss, 'batch_f1':f1}

    def training_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.train_best_f1.update(avg_f1)
        self.log('train/f1', avg_f1, prog_bar=True)
        self.log('train/best_f1', self.train_best_f1.compute(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, labels, pred_labels = self.shared_step(batch)
        f1 = self.metric(y_true=labels, y_pred=pred_labels)
        return {'loss':loss, 'batch_f1':f1}

    def validation_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.val_best_f1.update(avg_f1)
        self.log('val/f1', avg_f1, prog_bar=True)
        self.log('val/best_f1', self.val_best_f1.compute(), prog_bar=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]




