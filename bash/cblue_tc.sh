# ----------roberta val/best_f1 = 0.833----------------#
python run.py \
mode=exp \
name=cblue_tc \
trainer=ddp \
trainer.max_epochs=25 \
trainer.precision=16 \
datamodule=text_classification \
datamodule.dataset=cblue_tc_single \
datamodule.batch_size=96 \
datamodule.max_length=128 \
datamodule.pretrained_model=chinese-roberta-wwm-ext \
model=bert_text_classification \
model.dropout=0.2 \
model.lr=5e-5 \
model.hidden_size=256 \
model.weight_decay=0.0 \
seed=11

