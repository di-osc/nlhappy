import torch.nn as nn
import torch
from ...metrics.span import SpanTokenF1
from ...utils.make_model import PLMBaseModel
from ...layers import SimpleDense

class BertForQuestionAnswering(PLMBaseModel):
    def __init__(self,
                 lr: float = 3e-5,
                 scheduler: str = 'linear_warmup',
                 weight_decay: float = 0.01,
                 hidden_size: int = 256,
                 **kwargs) : 
        super().__init__()

        self.plm = self.get_plm_architecture()
        self.classifier = SimpleDense(input_size=self.plm.config.hidden_size, hidden_size=hidden_size, output_size=2)

        self.criterion = nn.CrossEntropyLoss()

        self.train_metric = SpanTokenF1()
        self.val_metric = SpanTokenF1()
        self.test_metric = SpanTokenF1()
        
        
    def setup(self, stage: str) -> None:
        self.trainer.datamodule.dataset.set_transform(self.trainer.datamodule.sequence_transform)


    def forward(self, input_ids, token_type_ids, attention_mask=None) -> torch.Tensor:
        x = self.plm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(x)
        return logits


    def shared_step(self, batch):
        tags = batch['tags']
        logits = self(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'])
        loss = self.criterion(logits.permute(1,2,0), tags.permute(1,0))
        preds = logits.argmax(dim=-1)
        return loss, preds, tags


    def training_step(self, batch, batch_idx):
        loss, preds, tags = self.shared_step(batch=batch)
        self.train_metric(preds, tags)
        self.log('train/f1', self.train_metric, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, preds, tags = self.shared_step(batch)
        self.val_metric(preds, tags)
        self.log('val/f1', self.val_metric, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        loss, preds, tags = self.shared_step(batch)
        self.test_metric(preds, tags)
        self.log('test/f1', self.test_metric, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler]


    def predict(self, text: str, question: str, device: str='cpu'):
        
        inputs = self.tokenizer(question,
                                text,
                                max_length=512,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
        inputs.to(device)
        logits = self(**inputs)
        preds = logits.argmax(dim=-1)
        return preds