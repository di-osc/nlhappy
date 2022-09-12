from ...utils.make_model import PLMBaseModel
from ...layers import LayerNorm, Biaffine
from ...metrics.entity import Entity, EntityF1
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List
from collections import deque, defaultdict



class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),            #常用于图像
            nn.Conv2d(input_size, channels, kernel_size=1),   #将通道数由552改为96
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)    #将三组卷积结果进行拼接
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)                                          #将权重变为正态分布
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1) #加偏执项
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)  #矩阵点积
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)    #MLP就是一个线性变化器+一个激活函数+dropout
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):           #x,y相同，都是每句话中每个字的特征,z是卷积得到的结果
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class W2NERForEntityExtraction(PLMBaseModel):
    """w2ner 统一解决嵌套非连续实体, 模型复现
    reference: 
        - https://github.com/ljynlp/W2NER
    """
    def __init__(self,
                 lr: float,
                 scheduler: str= 'linear_warmup_step',
                 weight_decay: float = 0.01,
                 conv_hidden_size: int = 96,
                 dilation: List[int] = [1,2,3],
                 dist_emb_size: int = 20,
                 reg_emb_size: int = 20,
                 biaffine_size: int = 512,
                 dropout: float = 0.2,
                 ffnn_hidden_size: int = 288,
                 **kwargs):
        super().__init__()
        self.plm = self.get_plm_architecture()

        self.distance_embeddings = nn.Embedding(20, self.hparams.dist_emb_size)
        self.region_embeddings = nn.Embedding(3, self.hparams.reg_emb_size)

        # self.lstm = nn.LSTM(self.plm.config.hidden_size, 
        #                        self.hparams.lstm_hidden_size, 
        #                        num_layers=1, 
        #                        batch_first=True,
        #                        bidirectional=True)

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.cln = LayerNorm(hidden_size=self.plm.config.hidden_size, 
                             conditional_size=self.plm.config.hidden_size)
        
        conv_input_size = self.hparams.dist_emb_size + self.hparams.reg_emb_size + self.plm.config.hidden_size
        self.convLayer = ConvolutionLayer(input_size=conv_input_size,
                                          channels=self.hparams.conv_hidden_size,
                                          dilation=self.hparams.dilation,
                                          dropout=self.hparams.dropout)

        self.predictor = CoPredictor(cls_num=len(self.hparams.label2id), 
                                     hid_size=self.plm.config.hidden_size, 
                                     biaffine_size=self.hparams.biaffine_size,
                                     channels=self.hparams.conv_hidden_size * len(self.hparams.dilation), 
                                     ffnn_hid_size=self.hparams.ffnn_hidden_size,
                                     dropout=self.hparams.dropout)

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.train_metric = EntityF1()
        self.val_metric = EntityF1()


    def forward(self, input_ids, token_type_ids, attention_mask, grid_mask, distance_ids):
        # 将bert的输出取后四层的平均 -> b, l, h
        hiddens = self.plm(input_ids, token_type_ids, attention_mask, output_hidden_states=True).hidden_states
        hidden = torch.stack(hiddens[-4:], dim=-1).mean(-1) 
        hidden = self.dropout(hidden)
        # 经过lstm -> b, l, h
        # packed = pack_padded_sequence(hidden, attention_mask.sum(dim=-1), batch_first=True, enforce_sorted=False)
        # packed, (_, _) = self.lstm(packed)
        # hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=attention_mask)
        # 经过cln -> b, l, l, h
        cln = self.cln(hidden.unsqueeze(2), hidden)
        # 将distance, region 和 cln输出 拼接 -> b,l,l,h_concat
        dis_emb = self.distance_embeddings(distance_ids)
        region_ids = self.get_region_ids(grid_mask)
        reg_emb = self.region_embeddings(region_ids)
        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        # 经过卷积层 -> b, l, l, h_ffnn
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask.eq(0).unsqueeze(-1), 0.0) #grid_mask2d根据句子长度对句子设置padding，即将conv_inputs中padding的位置用0填补
        conv_outputs = self.convLayer(conv_inputs)                      #将3个 Dilated Convolution的结果拼接输出
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask.eq(0).unsqueeze(-1), 0.0)
        # 输入预测层 -> b, l, l, num_labels
        logits = self.predictor(hidden, hidden, conv_outputs)

        return logits


    def get_region_ids(self, grid_mask):
        # tril_mask: batch seq seq
        tril_mask = torch.tril(grid_mask)
        region_ids = tril_mask + grid_mask
        return region_ids


    def get_grid_mask(self, attention_mask):
        grid_mask = torch.transpose(attention_mask.unsqueeze(1), 1, 2) * attention_mask.unsqueeze(1)
        return grid_mask


    def extract_ents(self, label_ids, attention_mask):
        class Node:
            def __init__(self):
                self.THW = []                # [(tail, type)]
                self.NNW = defaultdict(set)
        q = deque()
        ents = []
        length = torch.sum(attention_mask, dim=-1)
        for instance, l, in zip(label_ids, length):
            l = l.item()
            nodes = [Node() for _ in range(l)]
            predicts = []
            for cur in reversed(range(l)):
                heads = []
                for pre in range(cur+1):
                    if instance[cur, pre] > 1: 
                            nodes[pre].THW.append((cur, instance[cur, pre]))
                            heads.append(pre)
                        # NNW
                    if pre < cur and instance[pre, cur] == 1:
                        # cur node
                        for head in heads:
                            nodes[pre].NNW[(head,cur)].add(cur)
                        # post nodes
                        for head,tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                nodes[pre].NNW[(head,tail)].add(cur)
                for tail,type_id in nodes[cur].THW:
                        if cur == tail:
                            predicts.append(([cur], type_id))
                            continue
                        q.clear()
                        q.append([cur])
                        while len(q) > 0:
                            chains = q.pop()
                            for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                                if idx == tail:
                                    predicts.append((chains + [idx], type_id))
                                else:
                                    q.append(chains + [idx])
            preds = [Entity(label=l.item(), indexes=tuple(i)) for i,l in predicts]
            ents.append(set(preds))
        return ents
    def step(self, batch):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        distance_ids = batch['distance_ids']
        label_ids = batch['label_ids']
        grid_mask = self.get_grid_mask(attention_mask)
        logits = self(input_ids, token_type_ids, attention_mask, grid_mask, distance_ids)
        loss = self.criterion(logits.permute(0, 3, 1, 2), label_ids)
        loss = torch.sum(loss * grid_mask) / torch.sum(grid_mask) 
        return logits, loss

    
    def training_step(self, batch, batch_idx):
        # targs = batch['label_ids']
        # attention_mask = batch['attention_mask']
        # targs = self.extract_ents(targs, attention_mask)
        logits, loss = self.step(batch)
        # preds = logits.argmax(dim=-1)
        # preds = self.extract_ents(preds, attention_mask)
        # self.train_metric(preds, targs)
        # self.log('train/f1', self.train_metric, on_step=True, prog_bar=True)
        return {'loss':loss}


    def validation_step(self, batch, batch_idx):
        targs = batch['label_ids']
        attention_mask = batch['attention_mask']
        targs = self.extract_ents(targs, attention_mask)
        logits, loss = self.step(batch)
        preds = logits.argmax(dim=-1)
        preds = self.extract_ents(preds, attention_mask)
        self.val_metric(preds, targs)
        self.log('val/f1', self.val_metric, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        plm_params = set(self.plm.parameters())
        other_params = list(set(self.parameters()) - plm_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr,
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr * 10,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': self.hparams.lr * 10,
             'weight_decay': self.hparams.weight_decay},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler_config = self.get_scheduler_config(optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler_config]


    def predict(self, text):
        pass