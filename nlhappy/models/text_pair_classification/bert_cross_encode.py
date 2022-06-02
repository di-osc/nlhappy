import torch
from pytorch_lightning import LightningModule
from torchmetrics import F1Score
from typing import List, Any, Dict
from transformers import AutoModel, BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F
import os




class BERTCrossEncoder(LightningModule):
    '''
    交互式的bert句子对分类模型
    '''
    def __init__(
        self, 
        lr: float ,
        hidden_size: int,
        dropout: float,
        weight_decay: float ,
        **kwargs
        ):
        super(BERTCrossEncoder, self).__init__()  
        
        self.save_hyperparameters(logger=False)
        
        self.bert = BertModel(self.hparams['trf_config'])
        self.pooler = torch.nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, len(self.hparams.label2id))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_f1 = F1Score(average='macro', num_classes=len(self.hparams.label2id))
        self.val_f1 = F1Score(average='macro', num_classes=len(self.hparams.label2id))
        self.test_f1 = F1Score(average='macro', num_classes=len(self.hparams.label2id))
        self.tokenizer = self._init_tokenizer()
        

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x.last_hidden_state
        x = x * attention_mask.unsqueeze(-1).float()
        x = torch.sum(x, dim=1)
        x = x / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        x = self.pooler(x)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits


    def shared_step(self, batch):
        inputs = batch['inputs']
        labels = batch['label_ids']
        logits = self(**inputs)
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
        return {'loss': loss}


    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.test_f1(preds, labels)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}


    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]

    def _init_tokenizer(self):
        with open('./vocab.txt', 'w') as f:
            for k in self.hparams.vocab.keys():
                f.writelines(k + '\n')
        self.hparams.trf_config.to_json_file('./config.json')
        tokenizer = BertTokenizer.from_pretrained('./')
        os.remove('./vocab.txt')
        os.remove('./config.json')
        return tokenizer
    
    def to_onnx(self, file_path: str):
        text1 = '我是中国人'
        text2 = '我是河北人'
        torch_inputs = self.tokenizer(text1, text2, return_tensors='pt')
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
        
        
    def predict(self, text_pair: List, device: str='cpu') -> Dict[str, float]:
        device = torch.device(device)
        inputs = self.tokenizer(
                text_pair[0],
                text_pair[1],
                padding='max_length',
                max_length=self.hparams.max_length,
                return_tensors='pt',
                truncation=True)
        inputs.to(device)
        # self.to(device)
        # self.freeze()
        self.eval()
        with torch.no_grad():
            logits = self(**inputs)
            scores = torch.nn.functional.softmax(logits, dim=-1).tolist()
            cats = {}
            for i, v in enumerate(scores[0]): # scores : [[0.1, 0.2, 0.3, 0.4]]
                id2label = {v: k for k, v in self.hparams.label2id.items()}
                cats[id2label[i]] = v
        return sorted(cats.items(), key=lambda x: x[1], reverse=True)

