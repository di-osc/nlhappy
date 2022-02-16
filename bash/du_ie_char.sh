# --------------------------#
python run.py \
mode=exp \
monitor=val/f1 \
name=du_ie_char \
trainer=ddp \
trainer.max_epochs=25 \
trainer.precision=16 \
datamodule=triple_extraction \
datamodule.dataset=du_ie_char \
datamodule.batch_size=12 \
datamodule.max_length=302 \
datamodule.pretrained_model=chinese-roberta-wwm-ext \
model=bert_gplinker \
model.dropout=0.2 \
model.lr=3e-5 \
model.hidden_size=64 \
model.weight_decay=0.01 \
seed=11