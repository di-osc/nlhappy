#  roberta-wwm-ext ---------#

python run.py \
mode=exp \
name=cblue_ee_chartoken \
trainer=ddp \
trainer.max_epochs=20 \
trainer.precision=16 \
test_after_training=False \
datamodule=token_classification \
datamodule.dataset=cblue_ee_chartoken \
datamodule.batch_size=16 \
datamodule.max_length=420 \
datamodule.pretrained_model=chinese-roberta-wwm-ext \
model=bert_crf \
model.dropout=0.2 \
model.lr=3e-5 \
model.hidden_size=256 \
model.weight_decay=0.01 \
seed=22