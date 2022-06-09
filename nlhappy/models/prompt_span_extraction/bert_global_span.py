import pytorch_lightning as pl
from transformers import AutoModel,  AutoTokenizer
import torch.nn as nn
import torch
import os
from ...metrics.span import SpanF1
from ...layers import MultiLabelCategoricalCrossEntropy, EfficientGlobalPointer




class BERTGlobalSpan(pl.LightningModule):
    
    def __init__(self,
                hidden_size: int,
                lr: float,
                weight_decay: float,
                dropout: float,
                threshold: float = 0.5,
                **kwargs):
        """全局span抽取模型
        Args:
            hidden_size (int): 隐藏层维度
            lr (float): 学习率
            weight_decay (float): 权重衰减
            dropout (float): dropout比例
            threshold (float, optional): 抽取阈值. Defaults to 0.5.
            load_plm (bool, optional): 是否加载plm的参数. Defaults to False.
            kwargs: 其他参数包括trf_config, vocab
        """
        super().__init__()
        
        self.save_hyperparameters()   


        self.plm = AutoModel.from_config(self.hparams.trf_config)
        self.classifier = EfficientGlobalPointer(
                        input_size=self.plm.config.hidden_size, 
                        hidden_size=hidden_size,
                        output_size=1)

        self.dropout = nn.Dropout(dropout)
        self.criterion = MultiLabelCategoricalCrossEntropy()

        self.train_metric = SpanF1()
        self.val_metric = SpanF1()
        self.test_metric = SpanF1()
        self.tokenizer = self._init_tokenizer()

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        x = self.plm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x, mask=attention_mask)
        return logits



    def shared_step(self, batch):
        span_ids = batch['span_ids']
        inputs = batch['inputs']
        logits = self(**inputs)
        pred = logits.ge(self.hparams.threshold).float()
        batch_size, ent_type_size = logits.shape[:2]
        y_true = span_ids.reshape(batch_size*ent_type_size, -1)
        y_pred = logits.reshape(batch_size*ent_type_size, -1)
        loss = self.criterion(y_pred, y_true)
        return loss, pred, span_ids


    def training_step(self, batch, batch_idx):
        loss, pred, true = self.shared_step(batch)
        self.train_metric(pred, true)
        self.log('train/f1', self.train_metric, on_step=True, on_epoch=True, prog_bar=True)
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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [optimizer], [scheduler]


    def _init_tokenizer(self):
        with open('./vocab.txt', 'w') as f:
            for k in self.hparams.vocab.keys():
                f.writelines(k + '\n')
        self.hparams.trf_config.to_json_file('./config.json')
        tokenizer = AutoTokenizer.from_pretrained('./')
        os.remove('./vocab.txt')
        os.remove('./config.json')
        return tokenizer
    
    
    def predict(self, prompt: str, text: str, device: str='cpu', threshold = None):
        if threshold is None:
            threshold = self.hparams.threshold
        inputs = self.tokenizer(
            prompt,
            text,
            max_length=self.hparams.max_length,
            truncation=True,
            return_tensors='pt')
        offset_ls = self.tokenizer(
            prompt,
            text,
            max_length=self.hparams.max_length,
            truncation=True,
            return_offsets_mapping=True)['offset_mapping']
        
        inputs.to(device)
        logits = self(**inputs)
        spans_ls = torch.nonzero(logits>threshold).tolist()
        spans = []
        for span in spans_ls :
            start = span[2]
            end = span[3]
            start_text = offset_ls[start][0]
            end_text = offset_ls[end][1]
            spans.append([start_text, end_text, text[start_text:end_text]])
        return spans

    def to_onnx(self, file_path: str):
        prompt = '国家'
        text = '我是中国人'
        torch_inputs = self.tokenizer(prompt, text, return_tensors='pt')
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