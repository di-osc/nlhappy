nlhappy \
datamodule=text_classification \
datamodule.dataset=num \
datamodule.plm=chinese-roberta-wwm-ext \
datamodule.max_length=100 \
datamodule.batch_size=16 \
model=bert_text_classification \
model.lr=5e-5 \
model.hidden_size=768  \
model.dropout=0.1 \
model.weight_decay=0.00 \
seed=12345

# nlhappy \
# trainer=ddp \
# datamodule=span_classification \
# datamodule.dataset=CMeEE \
# datamodule.batch_size=14 \
# datamodule.max_length=420 \
# datamodule.plm=chinese-macbert-base \
# model=bert_global_pointer \
# model.dropout=0.2 \
# model.lr=2e-5 \
# model.hidden_size=256 \
# model.weight_decay=0.01 \
# seed=2222