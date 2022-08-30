import torch
from torchmetrics import F1Score
from typing import List, Any
from ...layers import SimpleDense, MultiDropout
from ...utils.make_model import PLMBaseModel
from typing import List, Dict
import torch.nn.functional as F

class BertTextClassification(PLMBaseModel):
    '''
    文本分类模型
    '''
    def __init__(self, 
                hidden_size: int ,
                scheduler: str,
                lr: float ,
                weight_decay: float ,
                dropout: float,
                **kwargs):
        super(BertTextClassification, self).__init__()  
        self.save_hyperparameters()

        # 模型架构
        self.bert = self.get_plm_architecture()
        self.classifier = SimpleDense(self.bert.config.hidden_size, hidden_size, len(self.hparams.label2id))
        self.dropout = MultiDropout()
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

        # 评价指标
        self.train_f1 = F1Score(num_classes=len(self.hparams.label2id), average='macro')
        self.val_f1= F1Score(num_classes=len(self.hparams.label2id), average='macro')
        self.test_f1 = F1Score(num_classes=len(self.hparams.label2id), average='macro')


    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0]
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
        self.log('train/loss',loss)
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
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr *5, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr * 5, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler_config = self.get_scheduler_config(optimizer, self.hparams.scheduler)
        return [optimizer], [scheduler_config]


    def predict(self, text: str, device: str='cpu') -> Dict[str, float]:
        device = torch.device(device)
        inputs = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.hparams.max_length,
                return_tensors='pt',
                truncation=True)
        inputs.to(device)
        self.eval()
        with torch.no_grad():
            logits = self(**inputs)
            scores = torch.nn.functional.softmax(logits, dim=-1).tolist()
            cats = {}
            for i, v in enumerate(scores[0]):   # scores : [[0.1, 0.2, 0.3, 0.4]]
                cats[self.hparams.id2label[i]] = v
        return sorted(cats.items(), key=lambda x: x[1], reverse=True)

    
    def to_onnx(self, file_path: str):
        text = '我是中国人'
        torch_inputs = self.tokenizer(text, return_tensors='pt')
        dynamic_axes = {
                    'input_ids': {0: 'batch', 1: 'seq'},
                    'attention_mask': {0: 'batch', 1: 'seq'},
                    'token_type_ids': {0: 'batch', 1: 'seq'},
                }
        with torch.no_grad():
            torch.onnx.export(
                model=self,
                args=tuple(torch_inputs.values()), 
                f=file_path, 
                input_names=list(torch_inputs.keys()),
                dynamic_axes=dynamic_axes, 
                opset_version=14,
                output_names=['logits'],
                export_params=True)
        print('export to onnx successfully')

