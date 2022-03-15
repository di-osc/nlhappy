import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy
from torchmetrics import MaxMetric
from typing import List, Any
from transformers import AutoModel
import torch.nn.functional as F




class BERTBiEncoder(LightningModule):
    '''双塔模型'''
    def __init__(
        self,
        bert_name: str,
        mid_size: int,
        output_size: int,
        lr: float,
        weight_decay: float
        ):
        super(BERTBiEncoder, self).__init__()
        self.save_hyperparameters(logger=False)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.hidden_layer = torch.nn.Linear(self.bert.config.hidden_size * 3, mid_size)
        self.classifier = torch.nn.Linear(mid_size, output_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()

    def forward(self, inputs_a, inputs_b):
        encoded_a = self.bert(**inputs_a).pooler_output
        encoded_b = self.bert(**inputs_b).pooler_output
        abs_diff = torch.abs(encoded_a - encoded_b)
        concat = torch.cat((encoded_a, encoded_b, abs_diff), dim=-1)
        hidden = F.relu(self.hidden_layer(concat))
        logits = self.classifier(hidden)
        return logits

    def shared_step(self, batch):
        inputs_a = batch['inputs_1']
        inputs_b = batch['inputs_2']
        labels = batch['label']
        logits = self(inputs_a, inputs_b)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        acc = self.train_acc(preds, labels)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'preds': preds, 'targets': labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        acc = self.val_acc(preds, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'preds': preds, 'targets': labels}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log('val/best_acc', self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        acc = self.test_acc(preds, labels)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'preds': preds, 'targets': labels}

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.log('lr', self.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]
    