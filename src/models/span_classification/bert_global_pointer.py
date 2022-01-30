import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
from ...metrics.span import SpanEvaluator
from torchmetrics import MaxMetric
from torch.utils.data import DataLoader
from ...utils.preprocessing import fine_grade_tokenize
from ..layers import GlobalPointer, MultiLabelCategoricalCrossEntropy, EfficientGlobalPointer

class BertGlobalPointer(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        weight_decay: float,
        dropout: float ,
        **data_params
    ) : 
        super().__init__()
        self.save_hyperparameters()

        self.label2id = data_params['label2id']
        self.id2label = {v:k for k,v in self.label2id.items()}

        self.bert = BertModel.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        # self.classifier = GlobalPointer(
        #     input_size=self.bert.config.hidden_size, 
        #     hidden_size=hidden_size, 
        #     output_size=len(self.label2id))
        # 使用更加高效的GlobalPointer https://kexue.fm/archives/8877
        self.classifier = EfficientGlobalPointer(
            input_size=self.bert.config.hidden_size, 
            hidden_size=hidden_size,
            output_size=len(self.label2id))

        self.tokenizer = BertTokenizer.from_pretrained(data_params['pretrained_dir'] + data_params['pretrained_model'])
        self.dropout = nn.Dropout(dropout)
        self.criterion = MultiLabelCategoricalCrossEntropy()
        self.metric = SpanEvaluator()
        self.train_best_f1 = MaxMetric()
        self.val_best_f1 = MaxMetric()



    def forward(self, inputs):
        x = self.bert(**inputs).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x, mask=inputs['attention_mask'])
        return logits


    def shared_step(self, batch):
        inputs, span_ids = batch['inputs'], batch['span_ids']
        logits = self(inputs)
        batch_size, ent_type_size = logits.shape[:2]
        y_true = span_ids.reshape(batch_size*ent_type_size, -1)
        y_pred = logits.reshape(batch_size*ent_type_size, -1)
        loss = self.criterion(y_pred, y_true)
        return logits, loss

    
    def training_step(self, batch, batch_idx):
        logits, loss = self.shared_step(batch)
        f1 = self.metric.get_sample_f1(logits, batch['span_ids'])
        return {'loss': loss, 'batch_f1': f1}

    def training_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.train_best_f1.update(avg_f1)
        self.log('train/f1', avg_f1, prog_bar=True)
        self.log('train/best_f1', self.train_best_f1.compute(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        pred, loss = self.shared_step(batch)
        f1 = self.metric.get_sample_f1(pred, batch['span_ids'])
        return {'loss': loss, 'batch_f1': f1}

    def validation_epoch_end(self, outputs):
        avg_f1 = torch.stack([x['batch_f1'] for x in outputs]).mean()
        self.val_best_f1.update(avg_f1)
        self.log('val/f1', avg_f1, prog_bar=True)
        self.log('val/best_f1', self.val_best_f1.compute(), prog_bar=True)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]


    def predict(self, text: str):
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens,
            is_pretokenized=True,
            add_special_tokens=True,
            return_tensors='pt')
        logits = self(inputs)
        spans_ls = torch.nonzero(logits>0).tolist()
        spans = []
        for span in spans_ls :
            start = span[2]
            end = span[3]
            spans.append([start-1, end-1, self.id2label[span[1]], text[start-1:end]])
        return spans



        

    


        


    

    
        
