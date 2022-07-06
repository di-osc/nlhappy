import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy
from typing import List, Any
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F
import os




class BERTBiEncoder(LightningModule):
    '''双塔模型'''
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        dropout: float,
        weight_decay: float,
        **kwargs
        ):
        super(BERTBiEncoder, self).__init__()
        self.save_hyperparameters(logger=False)
        self.bert = BertModel(self.hparams['bert_config'])
        self.dropout = torch.nn.Dropout(dropout)
        self.pooler = torch.nn.Linear(self.bert.config.hidden_size * 3, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, len(self.hparams.label2id))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=len(self.hparams.label2id))
        self.val_acc = Accuracy(num_classes=len(self.hparams.label2id))
        self.test_acc = Accuracy(num_classes=len(self.hparams.label2id))
        self.tokenizer = self._init_tokenizer()

    def forward(self, inputs_a, inputs_b):
        encoded_a = self.dropout(torch.mean(self.bert(**inputs_a).last_hidden_state, dim=1))
        encoded_b = self.dropout(torch.mean(self.bert(**inputs_b).last_hidden_state, dim=1))
        abs_diff = torch.abs(encoded_a - encoded_b)
        concat = torch.cat((encoded_a, encoded_b, abs_diff), dim=-1)
        hidden = F.relu(self.pooler(concat))
        logits = self.classifier(hidden)
        return logits

    def shared_step(self, batch):
        inputs_a = batch['inputs_a']
        inputs_b = batch['inputs_b']
        labels = batch['label_ids']
        logits = self(inputs_a, inputs_b)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels


    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.train_acc(preds, labels)
        self.log('train/acc',self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.val_acc(preds, labels)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.test_acc(preds, labels)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]

    def _init_tokenizer(self):
        with open('./vocab.txt', 'w') as f:
            for k in self.hparams.token2id.keys():
                f.writelines(k + '\n')
        self.hparams.bert_config.to_json_file('./config.json')
        tokenizer = BertTokenizer.from_pretrained('./')
        os.remove('./vocab.txt')
        os.remove('./config.json')
        return tokenizer
    