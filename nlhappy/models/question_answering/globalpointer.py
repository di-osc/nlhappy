import torch
from ...metrics.span import SpanF1
from ...utils.make_model import PLMBaseModel
from ...layers import MultiLabelCategoricalCrossEntropy, EfficientGlobalPointer


class GlobalPointerForQuestionAnswering(PLMBaseModel):
    def __init__(self,
                 lr: float,
                 scheduler: str = 'linear_warmup',
                 hidden_size: int = 64,
                 weight_decay: float = 0.01,
                 dropout: float = 0.2,
                 threshold: float = 0.5,
                 **kwargs) : 
        super().__init__()
        ## 手动optimizer 可参考https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#manual-optimization  

        self.plm = self.get_plm_architecture()
        self.classifier = EfficientGlobalPointer(input_size=self.plm.config.hidden_size, 
                                                 hidden_size=hidden_size,
                                                 output_size=1)

        self.criterion = MultiLabelCategoricalCrossEntropy()

        self.train_metric = SpanF1()
        self.val_metric = SpanF1()
        self.test_metric = SpanF1()
        
    def setup(self, stage: str) -> None:
        self.trainer.datamodule.dataset.set_transform(self.trainer.datamodule.gp_transform)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        x = self.plm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(x, mask=attention_mask)
        return logits

    def shared_step(self, batch):
        span_ids = batch['span_tags']
        logits = self(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'])
        pred = logits.ge(self.hparams.threshold).float()
        batch_size, ent_type_size = logits.shape[:2]
        y_true = span_ids.reshape(batch_size*ent_type_size, -1)
        y_pred = logits.reshape(batch_size*ent_type_size, -1)
        loss = self.criterion(y_pred, y_true)
        return loss, pred, span_ids


    def training_step(self, batch, batch_idx):
        loss, pred, targ = self.shared_step(batch=batch)
        self.train_metric(pred, targ)
        self.log('train/f1', self.train_metric, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, pred, true = self.shared_step(batch)
        self.val_metric(pred, true)
        self.log('val/f1', self.val_metric, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        loss, pred, true = self.shared_step(batch)
        self.test_metric(pred, true)
        self.log('test/f1', self.test_metric, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr * 5, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr * 5, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler]



    def predict(self, question: str, device: str='cpu', threshold = None):
        if threshold is None:
            threshold = self.hparams.threshold
        inputs = self.tokenizer(question,
                                max_length=self.hparams.max_length,
                                truncation=True,
                                return_tensors='pt')
        inputs.to(device)
        logits = self(**inputs)
        spans_ls = torch.nonzero(logits>threshold).tolist()
        spans = []
        for span in spans_ls :
            start = span[2]
            end = span[3]
            span_text = question[start-1:end]
            spans.append([start-1, end, self.hparams.id2label[span[1]], span_text])
        return spans