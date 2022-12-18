import torch
from ...metrics.span import SpanIndexF1
from ...utils.make_model import PLMBaseModel
from ...layers import SimpleDense, MultiLabelCategoricalCrossEntropy, MultiDropout
from ...data.doc import Doc
from typing import List, Set, Tuple

class PointerForQuestionAnswering(PLMBaseModel):
    def __init__(self,
                 lr: float = 3e-5,
                 scheduler: str = 'linear_warmup',
                 weight_decay: float = 0.01,
                 threshold: float = 0.0,
                 hidden_size: int = 256,
                 **kwargs) : 
        super().__init__()

        self.plm = self.get_plm_architecture()
        self.dropout = MultiDropout()
        self.start_classifier = SimpleDense(input_size=self.plm.config.hidden_size, hidden_size=hidden_size, output_size=1)
        self.end_classifier = SimpleDense(input_size=self.plm.config.hidden_size, hidden_size=hidden_size, output_size=1)

        self.criterion = MultiLabelCategoricalCrossEntropy()

        self.train_metric = SpanIndexF1()
        self.val_metric = SpanIndexF1()
        self.test_metric = SpanIndexF1()
        
        
    def setup(self, stage: str) -> None:
        self.trainer.datamodule.dataset.set_transform(self.trainer.datamodule.pointer_transform)


    def forward(self, input_ids, token_type_ids, attention_mask=None) -> torch.Tensor:
        x = self.plm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        start_logits = self.start_classifier(x)
        end_logits = self.end_classifier(x)
        return start_logits, end_logits


    def shared_step(self, batch):
        mask = batch['attention_mask']
        start_tags = batch['start_tags']
        end_tags = batch['end_tags']
        start_logits, end_logits = self(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'])
        start_loss = self.criterion(start_logits.squeeze(-1), start_tags)
        end_loss = self.criterion(end_logits.squeeze(-1), end_tags)
        loss = (start_loss + end_loss) / 2
        pred, _ = self.extract_spans(start_logits=start_logits, end_logits=end_logits, threshold=self.hparams.threshold)
        true, _ = self.extract_spans(start_logits=start_tags, end_logits=end_tags, threshold=self.hparams.threshold)
        return loss, pred, true
    
    def extract_spans(self, start_logits: torch.Tensor, end_logits: torch.Tensor, threshold: float) -> Tuple[List[Set], List[List[Tuple]]]:
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_logits = start_logits.chunk(start_logits.shape[0])
        end_logits = end_logits.chunk(end_logits.shape[0])
        spans = []
        batch_spans = []
        batch_indices = []
        for i in range(len(start_logits)):
            starts = torch.where(start_logits[i].squeeze(0) > threshold)[0].tolist()
            ends = torch.where(end_logits[i].squeeze(0) > threshold)[0].tolist()
            length = min(len(starts), len(ends))
            starts = starts[:length]
            ends = ends[:length]
            indices = set()
            spans = []
            for i in range(len(starts)):
                start = starts[i]
                end = ends[i]
                if end >= start:
                    spans.append((start, end+1))
                    for j in range(start, end+1):
                        indices.add(j)
            batch_indices.append(indices)
            batch_spans.append(spans)
        return batch_indices, batch_spans
            
            
    def training_step(self, batch, batch_idx):
        loss, pred, true = self.shared_step(batch=batch)
        self.train_metric(pred, true)
        self.log('train/f1', self.train_metric, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, pred, true = self.shared_step(batch)
        self.val_metric(pred, true)
        self.log('val/f1', self.val_metric, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        loss, preds, tags = self.shared_step(batch)
        self.test_metric(preds, tags)
        self.log('test/f1', self.test_metric, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in self.plm.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.plm.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.start_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.start_classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.end_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.end_classifier.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.hparams.lr, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        scheduler = self.get_scheduler_config(optimizer=optimizer, name=self.hparams.scheduler)
        return [optimizer], [scheduler]


    def predict(self, batch_question: List[str], batch_text: List[str], device: str='cpu'):
        inputs = self.tokenizer(batch_question,
                                batch_text,
                                max_length=512,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
        inputs.to(device)
        start_logits, end_logits = self(**inputs)
        _, batch_spans = self.extract_spans(start_logits=start_logits, end_logits=end_logits, threshold=self.hparams.threshold)
        align_batch_spans = []
        for spans in batch_spans:
            align_spans = []
            for span in spans:
                start = span[0]
                end = span[-1] - 1
                start_char_span= inputs.token_to_chars(start)
                end_char_span = inputs.token_to_chars(end)
                if start_char_span is not None and end_char_span is not None:
                    align_spans.append((start_char_span.start, end_char_span.end))
            align_batch_spans.append(align_spans)
        return align_batch_spans
    
    
    def set_annotation(self, doc: Doc, device: str = 'cpu', max_split_length: int = 350) -> Doc:
        for q, a in doc.questions.items():
            pieces = doc.split_by_sents(max_length=max_split_length)
            for piece in pieces:
                batch_text = [piece.text]
                batch_question = [q]
                spans = self.predict(batch_question=batch_question, batch_text=batch_text, device=device)[0]
                for span in spans:
                    answer_indices = piece.indices[span[0]: span[1]]
                    if len(answer_indices) > 0:
                        doc.add_answer_span(question=q, answer_indices=answer_indices)
        return doc  