nlhappy \
trainer=ddp \
datamodule=text_classification \
datamodule.dataset=TNEWS \
datamodule.plm=chinese-roberta-wwm-ext \
datamodule.max_length=100 \
datamodule.batch_size=16 \
model=bert_text_classification \
seed=1 