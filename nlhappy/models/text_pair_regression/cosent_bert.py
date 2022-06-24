import pytorch_lightning as pl 
from transformers import BertModel, AutoConfig, AutoTokenizer
import torch
import os
from torchmetrics import SpearmanCorrCoef
from ...layers.loss import CoSentLoss
from rich.progress import track



class CoSentBERT(pl.LightningModule):
    '''更好的句向量方案CoSent实现
    参考:
    - https://blog.csdn.net/HUSTHY/article/details/122821034?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-122821034-blog-124220249.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-122821034-blog-124220249.pc_relevant_aa&utm_relevant_index=2
    '''
    def __init__(self, 
                 lr: float,
                 weight_decay: float = 0.0,
                 **kwargs):
        """

        Args:
            lr (float): 学习率
            weight_decay (float): 权重衰减,默认为0
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.bert = BertModel(self.hparams.trf_config)
        self.tokenizer = self._init_tokenizer()
        self.criterion = CoSentLoss()
        self.train_meric = SpearmanCorrCoef()
        self.val_metric = SpearmanCorrCoef()
        self.test_metric = SpearmanCorrCoef()
        
    def forward(self, 
                input_ids, 
                token_type_ids, 
                attention_mask):
        hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hidden_state / attention_mask
        # return hidden_state[:,0]
    
    def step(self, batch):
        inputs_a = batch['inputs_a']
        inputs_b = batch['inputs_b']
        targs = batch['similarities']
        outputs_a = self(**inputs_a)
        outputs_b = self(**inputs_b)
        preds = torch.cosine_similarity(outputs_a, outputs_b)
        loss = self.criterion(preds, targs, self.device)
        return loss, preds, targs
    
    def training_step(self, batch, batch_idx):
        loss, preds, targs = self.step(batch)
        score = self.train_meric(preds, targs)
        self.log('train/spearman', self.train_meric, on_step=True, prog_bar=True)
        self.log('train/loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, targs = self.step(batch)
        score = self.val_metric(preds, targs)
        self.log('val/spearman', self.val_metric, on_epoch=True, prog_bar=True)
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
        device = torch.device(device)
        self.to(device)
        self.eval()
        with torch.no_grad():
            embed_a = self.tokenizer(text_a, padding='max_length', truncation=True, max_length=self.max_length)
            embed_b = self.tokenizer(text_b, padding='max_length', truncation=True, max_length=self.max_length)
            cos_sim = torch.cosine_similarity(embed_a, embed_b)
        return cos_sim