import pytorch_lightning as pl
from transformers import  AutoTokenizer, AutoConfig, BertModel
import torch
import os
from torchmetrics import SpearmanCorrCoef
from rich.progress import track

class SentenceBERT(pl.LightningModule):
    """文本表示模型sentence-bert
    参数:
    - lr: 学习率
    - weight_decay: 权重衰减
    - dropout: 随机失活
    """
    def __init__(self,
                lr: float ,
                weight_decay: float,
                **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.bert = BertModel(self.hparams.trf_config)
        self.tokenizer = self._init_tokenizer()
        self.criterion = torch.nn.MSELoss()
        self.train_meric = SpearmanCorrCoef()
        self.val_metric = SpearmanCorrCoef()
        self.test_metric = SpearmanCorrCoef()


    def forward(self, 
                input_ids, 
                token_type_ids, 
                attention_mask=None):
        hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hidden_state / attention_mask
        

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
        self.log('val/spearman', self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        loss, preds, targs = self.step(batch)
        self.test_metric(preds, targs)
        self.log('test/spearman', self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
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
    
    def encode(self, texts: list, device: str = 'cpu', batch_size: int = 500):
        device = torch.device(device)
        self.to(device)
        self.eval()
        all_embeds = torch.tensor([], device=device)
        with torch.no_grad():
            for i in track(range(0, len(texts), batch_size), description='Encoding'):
                batch = self.tokenizer(texts[i:i+batch_size], padding='max_length', return_tensors='pt', max_length=self.hparams.max_length, truncation=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                embeds = self(**batch)
                all_embeds = torch.cat((all_embeds, embeds), dim=0)
        return all_embeds
    
    
    def predict(self, text_a: str, text_b: str, device: str = 'cpu'):
        """得到两个文本的相似度
        参数:
        - text_a: a文本
        - text_b: b文本
        - device: 设备
        """
        device = torch.device(device)
        self.to(device)
        self.eval()
        with torch.no_grad():
            inputs_a = self.tokenizer(text_a, padding='max_length', truncation=True, max_length=self.max_length)
            inputs_b = self.tokenizer(text_b, padding='max_length', truncation=True, max_length=self.max_length)
            embed_a = self(**inputs_a)
            embed_b = self(**inputs_b)
            cos_sim = torch.cosine_similarity(embed_a, embed_b)
        return cos_sim       