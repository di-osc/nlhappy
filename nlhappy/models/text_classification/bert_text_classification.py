import torch
from pytorch_lightning import LightningModule
from torchmetrics import F1Score
from torchmetrics import MaxMetric
from typing import List, Any
from transformers import BertModel, BertTokenizer
from ...layers.classifier import SimpleDense
from ...utils.preprocessing import fine_grade_tokenize
from typing import List, Dict
from datasets import Dataset, concatenate_datasets

class BertTextClassification(LightningModule):
    '''
    文本分类模型
    '''
    def __init__(
        self, 
        hidden_size: int ,
        lr: float ,
        weight_decay: float ,
        dropout: float,
        **data_params
        ):
        super().__init__()  
        self.save_hyperparameters()

        self.label2id = self.hparams['label2id']
        self.id2label = {v:k for k,v in self.label2id.items()}
        self.tokenizer = BertTokenizer.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        
        #模型架构
        self.encoder = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.classifier = SimpleDense(self.encoder.config.hidden_size, hidden_size, len(self.label2id))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

        # 评价指标
        self.train_f1 = F1Score(num_classes=len(self.label2id), average='macro')
        self.val_f1= F1Score(num_classes=len(self.label2id), average='macro')
        self.test_f1 = F1Score(num_classes=len(self.label2id), average='macro')


    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output
        logits = self.classifier(x)  # (batch_size, output_size)
        return logits
        
    def shared_step(self, batch):
        inputs, label_ids = batch['inputs'], batch['label_ids']
        logits = self(**inputs)
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]


    def predict(self, text: str, device: str) -> Dict[str, float]:
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
                tokens,
                padding='max_length',
                max_length=self.hparams.max_length,
                return_tensors='pt',
                truncation=True)
        inputs.to(device)
        logits = self(inputs)
        scores = torch.nn.functional.softmax(logits, dim=-1).tolist()
        cats = {}
        for i, v in enumerate(scores[0]):   # scores : [[0.1, 0.2, 0.3, 0.4]]
            cats[self.id2label[i]] = v
        
        return cats

