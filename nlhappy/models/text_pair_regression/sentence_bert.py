import pytorch_lightning as pl
from transformers import  AutoTokenizer, AutoConfig, BertModel
import torch
import os
from torchmetrics import SpearmanCorrCoef

class SentenceBERT(pl.LightningModule):
    """sentence-bert
    参数:
    - """
    def __init__(self,
                lr: float ,
                weight_decay: float,
                dropout: float=0.5,
                **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.bert = BertModel(self.hparams.trf_config)
        self.dropout = torch.nn.Dropout(dropout)
        self.tokenizer = self._init_tokenizer()
        self.criterion = torch.nn.MSELoss()
        self.train_meric = SpearmanCorrCoef()
        self.val_metric = SpearmanCorrCoef()


    def forward(self, 
                input_ids, 
                token_type_ids, 
                attention_mask=None):
        x = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        # x.shape =  batch_size, seq_len, hidden_size
        x= torch.mean(x, dim=1)
        return x

    def on_train_start(self) -> None:
        ckpt_path = os.path.join(self.hparams.plm_dir, self.hparams.plm, 'pytorch_model.bin')
        state_dict = torch.load(ckpt_path)
        self.bert.load_state_dict(state_dict)
    
    def step(self, batch):
        inputs_a = batch['inputs_a']
        inputs_b = batch['inputs_b']
        targs = batch['similarities']
        outputs_a = self(**inputs_a)
        outputs_b = self(**inputs_b)
        preds = torch.cosine_similarity(outputs_a, outputs_b)
        loss = self.criterion(preds, targs)
        return loss, preds, targs

    def training_step(self, batch, batch_idx):
        loss, preds, targs = self.step(batch)
        self.train_meric(preds, targs)
        self.log('train/spearman', self.train_meric, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, targs = self.step(batch)
        self.val_metric(preds, targs)
        self.log('val/spearman', self.val_metric, on_step=True, on_epoch=True, prog_bar=True)
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
        tokenizer = AutoTokenizer.from_pretrained('./')
        os.remove('./vocab.txt')
        os.remove('./config.json')
        return tokenizer