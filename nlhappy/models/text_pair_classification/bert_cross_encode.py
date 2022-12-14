import torch
from torchmetrics import F1Score
from typing import List, Any, Dict
import torch.nn.functional as F
from ...layers import SimpleDense, MultiDropout
from ...utils.make_model import PLMBaseModel



class BERTCrossEncoder(PLMBaseModel):
    '''
    交互式的bert句子对分类模型
    '''
    def __init__(self, 
                 lr: float ,
                 hidden_size: int,
                 weight_decay: float ,
                 scheduler: str = 'linear_warmup',
                 **kwargs):
        super().__init__()  
        
        # 模型结构
        self.bert = self.get_plm_architecture(add_pooler_layer=True)
        self.dropout = MultiDropout()
        self.classifier = SimpleDense(self.bert.config.hidden_size, hidden_size=hidden_size, output_size=len(self.hparams.label2id))
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        # 指标评价函数
        self.train_f1 = F1Score(num_classes=len(self.hparams.label2id))
        self.val_f1 = F1Score(num_classes=len(self.hparams.label2id))
        self.test_f1 = F1Score(num_classes=len(self.hparams.label2id))
        # 预处理流程   
        
    def setup(self, stage: str) -> None:
        self.trainer.datamodule.dataset.set_transform(self.trainer.datamodule.cross_transform)     

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def shared_step(self, batch):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label_ids']
        logits = self(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.train_f1(preds, labels)
        self.log('train/f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.val_f1(preds, labels)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.test_f1(preds, labels)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr *5, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr *5, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler]
        
        
    def predict(self, batch_text_a: List[str], batch_text_b: List[str], device: str='cpu') -> Dict[str, float]:
        device = torch.device(device)
        inputs = self.tokenizer(batch_text_a,
                                batch_text_b,
                                padding=True,
                                max_length=512,
                                return_tensors='pt',
                                truncation=True)
        inputs.to(device)
        self.eval()
        with torch.no_grad():
            logits = self(**inputs)
            label_ids = logits.argmax(dim=-1).tolist()
            labels = [self.hparams.id2label[i] for i in label_ids]
        return labels