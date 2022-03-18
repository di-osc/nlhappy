import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from ...layers import SimpleDense
import torch.nn as nn
from torchmetrics import F1Score
import torch


class BertTextMultiClassification(pl.LightningModule):
    """基于bert的多标签文本分类"""
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        dropout: float,
        weight_decay: float,
        threshold: float = 0.5,
        **data_params
    ):
        super(BertTextMultiClassification, self).__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(data_params['label2id']))

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_f1 = F1Score(num_classes=len(data_params['label2id']), average='macro', multiclass=True, mdmc_average='global')
        self.val_f1 = F1Score(num_classes=len(data_params['label2id']), average='macro', multiclass=True, mdmc_average='global')
        self.test_f1 = F1Score(num_classes=len(data_params['label2id']), average='macro', multiclass=True, mdmc_average='global')

        self.tokenizer = BertTokenizer.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])

    def forward(self, input_ids, token_type_ids, attention_mask):
        pooled = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


    def shared_step(self, batch):
        inputs, label_ids = batch['inputs'], batch['label_ids']
        logits = self(**inputs)
        loss = self.criterion(logits, label_ids)
        pred_ids = logits.sigmoid().ge(self.hparams.threshold).long()
        return loss,  pred_ids, label_ids

    def training_step(self, batch, batch_id):
        loss, pred_ids, label_ids = self.shared_step(batch)
        self.train_f1(pred_ids, label_ids.long())
        self.log('train/f1', self.train_f1, on_step=True, on_epoch=True,  prog_bar=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_id):
        loss, pred_ids, label_ids = self.shared_step(batch)
        self.val_f1(pred_ids, label_ids.long())
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True,  prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_id):
        loss, pred_ids, label_ids = self.shared_step(batch)
        self.test_f1(pred_ids, label_ids.long())
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True,  prog_bar=True)
        return {'test_loss': loss}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [optimizer], [scheduler]
