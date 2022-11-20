from ...layers import CRF, SimpleDense
from ...metrics.chunk import ChunkF1, get_entities
from ...utils.make_model import PLMBaseModel
import torch

class CRFForEntityExtraction(PLMBaseModel):
    def __init__(self,
                 lr: float,
                 hidden_size: int,
                 weight_decay: float,
                 scheduler: str = 'linear_warmup',
                 **kwargs):
        super().__init__()

        self.plm = self.get_plm_architecture()

        self.classifier = SimpleDense(input_size=self.plm.config.hidden_size, 
                                      hidden_size=hidden_size, 
                                      output_size=len(self.hparams.label2id))

        self.crf = CRF(len(self.hparams.label2id))

        self.train_f1 = ChunkF1()
        self.val_f1 = ChunkF1()
        self.test_f1 = ChunkF1()

        
    def forward(self, input_ids, attention_mask, label_ids=None):
        x = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.classifier(x)
        mask = attention_mask.gt(0)
        if label_ids is not None :
            loss = self.crf(emissions, label_ids, mask=mask) * (-1)
            pred_ids= self.crf.decode(emissions, mask=mask)
            return loss, pred_ids
        else :
            return self.crf.decode(emissions, mask=mask)


    def shared_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label_ids = batch['tag_ids']
        #将label padding部分改为-1 
        label_ids[label_ids==-100] = -1
        loss, pred_ids = self(input_ids=input_ids, attention_mask=attention_mask,  label_ids=label_ids)
        pred_labels = []
        for ids in pred_ids:
            pred_labels.append([self.hparams.id2label[id] for id in ids])

        true_labels = []
        for i in range(len(label_ids)):
            indice = torch.where(label_ids[i] >= 0)
            ids = label_ids[i][indice].tolist()
            true_labels.append([self.hparams.id2label[id] for id in ids])
        return loss, pred_labels, true_labels
        

    def training_step(self, batch, batch_idx):
        loss, pred_labels, true_labels = self.shared_step(batch)
        self.train_f1(pred_labels, true_labels)
        self.log('train/f1', self.train_f1, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss':loss}


    def validation_step(self, batch, batch_idx):
        loss, pred_labels, true_labels = self.shared_step(batch)
        self.val_f1(pred_labels, true_labels)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss':loss}

    
    def test_step(self, batch, batch_idx):
        loss, pred_labels, true_labels = self.shared_step(batch)
        self.test_f1(pred_labels, true_labels)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss':loss}


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
             'lr': self.hparams.lr * 5, 'weight_decay': 0.0},
            {'params': self.crf.parameters(), 'lr': self.hparams.lr * 500, 'weight_decay': self.hparams.weight_decay}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler_config = self.get_scheduler_config(optimizer, self.hparams.scheduler)
        return [optimizer], [scheduler_config]


    def predict(self, text: str, device):
        inputs = self.tokenizer(text,
                                return_tensors='pt')
        inputs.to(device)
        outputs = self(**inputs)
        labels = [self.hparams.id2label[id] for id in outputs[0]]
        ents = get_entities(seq=labels[1:-1])    #去掉cls sep 位置
        new_ents = []
        for ent in ents:
            new_ents.append([ent[1], ent[2], ent[0], text[ent[1]:ent[2]+1]])
        return new_ents