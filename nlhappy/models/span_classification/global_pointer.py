import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
import os
from ...metrics.span import SpanF1
from ...utils.preprocessing import fine_grade_tokenize
from ...layers import MultiLabelCategoricalCrossEntropy, EfficientGlobalPointer
from ...tricks.adversarial_training import adversical_tricks




class GlobalPointer(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        lr: float,
        weight_decay: float,
        dropout: float,
        threshold: float = 0.5,
        adv: str =None,
        load_plm_checkpoint: bool = True,
        **kwargs
    ) : 
        super().__init__()
        self.save_hyperparameters()

        ## 手动optimizer 可参考https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#manual-optimization
        self.automatic_optimization = False    


        self.bert = BertModel(self.hparams.bert_config)
        self.classifier = EfficientGlobalPointer(
                        input_size=self.bert.config.hidden_size, 
                        hidden_size=hidden_size,
                        output_size=len(self.hparams.label2id)
                        )

        self.dropout = nn.Dropout(dropout)
        self.criterion = MultiLabelCategoricalCrossEntropy()

        self.train_metric = SpanF1()
        self.val_metric = SpanF1()
        self.test_metric = SpanF1()
        self.tokenizer = self._init_tokenizer()

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        logits = self.classifier(x, mask=attention_mask)
        return logits


    def on_train_start(self) -> None:
        if self.hparams.load_plm_checkpoint:
            state_dict = torch.load(self.hparams.plm_dir + self.hparams.plm + '/pytorch_model.bin')
            self.bert.load_state_dict(state_dict)
        if self.hparams.adv :
            self.adv = adversical_tricks.get(self.hparams.adv)(self.bert)


    def shared_step(self, batch):
        span_ids = batch['span_ids']
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


    def _init_tokenizer(self):
        with open('./vocab.txt', 'w') as f:
            for k in self.hparams.token2id.keys():
                f.writelines(k + '\n')
        self.hparams.bert_config.to_json_file('./config.json')
        tokenizer = BertTokenizer.from_pretrained('./')
        os.remove('./vocab.txt')
        os.remove('./config.json')
        return tokenizer


    def predict(self, text: str, device: str='cpu', threshold = None):
        if threshold is None:
            threshold = self.hparams.threshold
        tokens = fine_grade_tokenize(text, self.tokenizer)
        inputs = self.tokenizer.encode_plus(
            tokens,
            add_special_tokens=True,
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
            span_text = text[start-1:end]
            spans.append([start-1, end, self.hparams.id2label[span[1]], span_text])
        return spans

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
            
        
        
    


        


    

    
        
