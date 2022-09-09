import torch.nn as nn
import torch
from ...metrics.span import SpanF1
from ...utils.make_model import PLMBaseModel
from ...layers import MultiLabelCategoricalCrossEntropy, EfficientGlobalPointer, MultiDropout
from ...tricks.adversarial_training import adversical_tricks




class GlobalPointer(PLMBaseModel):
    def __init__(self,
                 lr: float,
                 hidden_size: int=64,
                 weight_decay: float=0.01,
                 dropout: float=0.2,
                 threshold: float = 0.5,
                 adv: str =None,
                 **kwargs) : 
        super().__init__()
        ## 手动optimizer 可参考https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#manual-optimization
        self.automatic_optimization = False    


        self.bert = self.get_plm_architecture()
        self.classifier = EfficientGlobalPointer(input_size=self.bert.config.hidden_size, 
                                                 hidden_size=hidden_size,
                                                 output_size=len(self.hparams.label2id))

        self.dropout = MultiDropout()
        self.criterion = MultiLabelCategoricalCrossEntropy()

        self.train_metric = SpanF1()
        self.val_metric = SpanF1()
        self.test_metric = SpanF1()

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x, mask=attention_mask)
        return logits


    def on_train_start(self) -> None:
        if self.hparams.adv :
            self.adv = adversical_tricks.get(self.hparams.adv)(self.bert)


    def shared_step(self, batch):
        span_ids = batch['label_ids']
        logits = self(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'])
        pred = logits.ge(self.hparams.threshold).float()
        batch_size, ent_type_size = logits.shape[:2]
        y_true = span_ids.reshape(batch_size*ent_type_size, -1)
        y_pred = logits.reshape(batch_size*ent_type_size, -1)
        loss = self.criterion(y_pred, y_true)
        return loss, pred, span_ids


    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        scheduler = self.lr_schedulers()
        loss, pred, true = self.shared_step(batch)
        self.manual_backward(loss)
        if self.hparams.adv == 'FGM':
            self.adv.attack()
            loss_adv,  _,  _ = self.shared_step(batch)
            self.manual_backward(loss_adv)
            self.adv.restore()
            self.log_dict({'train_loss': loss, 'adv_loss': loss_adv}, prog_bar=True)
        elif self.hparams.adv == "PGD":
            self.adv.backup_grad()
            K=3
            for t in range(K):
                self.adv.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    self.zero_grad()
                else:
                    self.adv.restore_grad()
                loss_adv, _, _ = self.shared_step(batch)
                self.manual_backward(loss_adv) # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.adv.restore()
            self.log_dict({'train_loss': loss, 'adv_loss': loss_adv}, prog_bar=True)
        optimizer.step()
        if self.trainer.is_last_batch:
            scheduler.step()
        self.train_metric(pred, true)
        self.log('train/f1', self.train_metric, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({'train_loss': loss}, prog_bar=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        loss, pred, true = self.shared_step(batch)
        self.val_metric(pred, true)
        self.log('val/f1', self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}


    def test_step(self, batch, batch_idx):
        loss, pred, true = self.shared_step(batch)
        self.test_metric(pred, true)
        self.log('test/f1', self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr * 5, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr * 5, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [optimizer], [scheduler]



    def predict(self, text: str, device: str='cpu', threshold = None):
        if threshold is None:
            threshold = self.hparams.threshold
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.hparams.max_length,
            truncation=True,
            return_tensors='pt')
        mapping = self.tokenizer(text,
                                 add_special_tokens=True,
                                 max_length=self.hparams.max_length,
                                 truncation=True,)
        inputs.to(device)
        logits = self(**inputs)
        spans_ls = torch.nonzero(logits>threshold).tolist()
        spans = []
        for span in spans_ls :
            start = span[2]
            end = span[3]
            span_text = text[start-1:end]
            spans.append([start-1, end, self.hparams.id2label[span[1]], span_text])
        return spans