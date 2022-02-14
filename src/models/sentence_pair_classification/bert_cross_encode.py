import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy
from torchmetrics import MaxMetric
from typing import List, Any
from transformers import AutoModel
import torch.nn.functional as F




class BERTCrossEncoder(LightningModule):
    '''
    交互式的bert句子对分类模型
    '''
    def __init__(
        self, 
        bert_name : str, 
        mid_size: int,
        output_size: int,
        lr: float ,
        weight_decay: float 
        ):
        super(BERTCrossEncoder, self).__init__()  
        
        self.save_hyperparameters(logger=False)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.classifier = torch.nn.Linear(mid_size, output_size)
        self.hidden_layer = torch.nn.Linear(self.bert.config.hidden_size, mid_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()
        self.optimizer = None
        self.scheduler = None
        

    def forward(self, inputs):
        x = self.bert(**inputs).last_hidden_state
        x = x.mean(dim=1)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits


    def shared_step(self, batch):
        inputs = batch['inputs']
        labels = batch['label']
        logits = self(inputs)
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

