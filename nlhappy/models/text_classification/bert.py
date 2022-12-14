import torch
from torchmetrics import F1Score
from typing import List, Tuple
from ...layers import SimpleDense, MultiDropout
from ...utils.make_model import PLMBaseModel


class BertForTextClassification(PLMBaseModel):
    '''
    文本分类模型
    '''
    def __init__(self, 
                 lr: float = 3e-5,
                 hidden_size: int = 256,
                 scheduler: str = 'linear_warmup',
                 weight_decay: float = 0.01,
                 **kwargs):
        super().__init__()  

        # 模型架构
        self.bert = self.get_plm_architecture(add_pooler_layer=True)
        self.classifier = SimpleDense(input_size=self.bert.config.hidden_size, 
                                      hidden_size=hidden_size, 
                                      output_size=len(self.hparams.id2label))
        self.dropout = MultiDropout()
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

        # 评价指标
        self.train_f1 = F1Score(num_classes=len(self.hparams.id2label), average='macro')
        self.val_f1= F1Score(num_classes=len(self.hparams.id2label), average='macro')
        self.test_f1 = F1Score(num_classes=len(self.hparams.id2label), average='macro')
        
    def setup(self, stage: str):
        self.trainer.datamodule.dataset.set_transform(self.trainer.datamodule.bert_transform)


    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x.pooler_output
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch_size, output_size)
        return logits
    
        
    def shared_step(self, batch):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        label_ids = batch['label_ids']
        logits = self(input_ids, token_type_ids, attention_mask)
        loss = self.criterion(logits, label_ids)
        pred_ids = torch.argmax(logits, dim=-1)
        return loss, pred_ids, label_ids

    def training_step(self, batch, batch_idx):
        loss, pred_ids, label_ids = self.shared_step(batch)
        self.train_f1(pred_ids, label_ids)
        self.log('train/f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        loss, pred_ids, label_ids = self.shared_step(batch)
        self.val_f1(pred_ids, label_ids)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}


    def test_step(self, batch, batch_idx):
        loss, pred_ids, label_ids = self.shared_step(batch)
        self.test_f1(pred_ids, label_ids)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler_config = self.get_scheduler_config(optimizer, self.hparams.scheduler)
        return [optimizer], [scheduler_config]


    def predict(self, text: str, device: str='cpu') -> List[Tuple[str, int]]:
        device = torch.device(device)
        inputs = self.tokenizer(text,
                                padding='max_length',
                                max_length=self.hparams.max_length,
                                return_tensors='pt',
                                truncation=True)
        self.to(device)
        inputs.to(device)
        self.eval()
        with torch.no_grad():
            logits = self(**inputs)
            scores = torch.nn.functional.softmax(logits, dim=-1).tolist()
            cats = {}
            for i, v in enumerate(scores[0]):   # scores : [[0.1, 0.2, 0.3, 0.4]]
                cats[self.hparams.id2label[i]] = v
        return sorted(cats.items(), key=lambda x: x[1], reverse=True)