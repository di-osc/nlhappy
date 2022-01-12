import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer, BertForPreTraining
from datasets import load_metric
import torch
import torch.nn.functional as F
import os
from ...metrics.chunk import f1_score


class BertSoftmax(pl.LightningModule):
    def __init__(self,
                 bert_name: str,
                 hidden_size: int,
                 lr: float,
                 weight_decay: float,
                 label2id: dict,
                 dropout:float =0.5 
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.label2id = label2id
        self.id2label = {id: label for label, id in self.label2id.items()}
        

        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = torch.nn.Dropout(dropout)
        self.hidden_layer = torch.nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(hidden_size, len(label2id))
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.metric = load_metric('seqeval')
        self.metric = f1_score

        self.train_labels = []
        self.train_pred_labels = []
        self.valid_labels = []
        self.valid_pred_labels = []
        
        
    
    def forward(self, inputs):
        x = self.bert(**inputs).last_hidden_state
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.classifier(x)
        return x

    def shared_step(self, batch):
        inputs = batch['inputs']
        label_ids = batch['label_ids']
        logits = self(inputs)
        mask = torch.gt(label_ids, -1)
        loss = self.criterion(logits.view(-1, len(self.label2id))[mask.view(-1)], label_ids.view(-1)[mask.view(-1)])
        pred_ids = torch.argmax(logits, dim=-1)
        labels = []
        pred_labels = []
        for i in range(len(label_ids)):
            indice = torch.where(label_ids[i] >= 0)
            ids = label_ids[i][indice].tolist()
            labels.append([self.id2label[id] for id in ids])
            pred = pred_ids[i][indice].tolist()
            pred_labels.append([self.id2label[id] for id in pred])
        return loss, labels, pred_labels



    def training_step(self, batch, batch_idx):
        loss, labels, pred_labels = self.shared_step(batch)
        # self.metric.add_batch(predictions=pred_labels,references=labels)
        self.train_labels.extend(labels)
        self.train_pred_labels.extend(pred_labels)
        return {'labels': labels, 'pred_labels':pred_labels, 'loss':loss}

    def training_epoch_end(self, outputs):
        f1 = self.metric(y_true=self.train_labels, y_pred=self.train_pred_labels)
        self.log('train/f1', f1, prog_bar=True)
        self.train_labels = []
        self.train_pred_labels = []

    def validation_step(self, batch, batch_idx):
        loss, labels, pred_labels = self.shared_step(batch)
        # self.metric.add_batch(predictions=pred_labels,references=labels)
        self.valid_labels.extend(labels)
        self.valid_pred_labels.extend(pred_labels)
        return {'labels': labels, 'pred_labels':pred_labels, 'loss':loss}

    def validation_epoch_end(self, outputs):
        f1 = self.metric(y_true=self.valid_labels, y_pred=self.valid_pred_labels)
        self.log('val/f1', f1, prog_bar=True)
        self.valid_labels = []
        self.valid_pred_labels = []

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        return [self.optimizer], [self.scheduler]





        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def prepare_data(self):
    #     '''下载数据集'''
    #     if not os.path.exists(self.hparams.data_dir + self.file_name):
    #         self.storage.download(filename = self.file_name, 
    #                             localfile = self.hparams.data_dir + self.file_name)
    #         with zipfile.ZipFile(file=self.hparams.data_dir + self.file_name, mode='r') as zf:
    #             zf.extractall(path=self.hparams.data_dir)
    #         os.remove(path=self.hparams.data_dir + self.file_name)


    # def set_transform(self, example):
    #     tokens = exmaple['tokens'][0]
    #     inputs = tokenizer(tokens, is_split_into_words=True, padding='max_length', max_length=self.max_length)
    #     inputs = dict(zip(inputs.keys(), map(torch.tensor, inputs.values())))
    #     labels = example['labels'][0]
    #     labels = self.label_pad_id + [self.label2id[label] for label in labels] 
    #     labels = labels + (self.max_length - len(labels)) * [self.label_pad_id]
    #     labels = torch.tensor(labels)
    #     return {'inputs'[inputs], 'label_ids'[labels]}



    # def setup(self):
    #     data = load_from_disk(self.hparams.data_dir + self.data_name)
    #     data.set_transform(transform=self.set_transform)
    #     self.train_dataset = data['train']
    #     self.valid_dataset = data['validation']
    #     self.label2id = srsly.read_json(self.hparams.data_dir + data_name + '/label2id.json')
    #     self.id2label = {v: k for k, v in self.label2id.items()}


    # def train_dataloader(self):
    #     return DataLoader(dataset=self.train_dataset, 
    #                       batch_size=self.hparams.batch_size, 
    #                       shuffle=True,
    #                       pin_memory=self.hparams.pin_memory,
    #                       num_workers=self.hparams.num_workers)


    # def val_dataloader(self):
    #     return DataLoader(dataset=self.valid_dataset,
    #                       batch_size=self.hparams.batch_size,
    #                       shuffle=False,
    #                       pin_memory=self.hparams.pin_memory,
    #                       num_workers=self.hparams.num_workers) 



