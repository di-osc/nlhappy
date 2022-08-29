import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
import os
from ...metrics.chunk import ChunkF1
from ...layers import SimpleDense


class BertTokenClassification(pl.LightningModule):
    def __init__(self,
                 hidden_size: int,
                 lr: float,
                 weight_decay: float,
                 dropout:float ,
                 **data_params):
        super().__init__()
        self.save_hyperparameters()
        self.label2id = data_params['label2id']
        self.id2label = {id: label for label, id in self.label2id.items()}
        

        self.bert = BertModel(data_params['bert_config'])
        self.classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(self.label2id))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_f1 = ChunkF1()
        self.val_f1 = ChunkF1()
        self.test_f1 = ChunkF1()

        
        
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_hidden_state = self.bert(input_ids, token_type_ids, attention_mask).last_hidden_state
        logits = self.classifier(bert_hidden_state)
        return logits

    def shared_step(self, batch):
        inputs = batch['inputs']
        label_ids = batch['label_ids']
        logits = self(**inputs)
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


    def on_train_start(self) -> None:
        state_dict = torch.load(self.hparams.pretrained_dir + self.hparams.plm + '/pytorch_model.bin')
        self.bert.load_state_dict(state_dict)


    def training_step(self, batch, batch_idx):
        loss, labels, pred_labels = self.shared_step(batch)
        self.train_f1(true=labels, pred=pred_labels)
        self.log('train/f1',self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss':loss}


    def validation_step(self, batch, batch_idx):
        loss, labels, pred_labels = self.shared_step(batch)
        self.val_f1(true=labels, pred=pred_labels)
        self.log('val/f1',self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss':loss}


    def test_step(self, batch, batch_idx):
        loss, labels, pred_labels = self.shared_step(batch)
        self.test_f1(true=labels, pred=pred_labels)
        self.log('test/f1',self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss':loss}

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]




