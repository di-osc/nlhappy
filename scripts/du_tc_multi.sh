# ----------roberta val/best_f1 = 0.833----------------#
python run.py \
mode=exp \
name=du_tc_multi \
trainer=ddp \
trainer.max_epochs=25 \
trainer.precision=16 \
datamodule=text_classification \
datamodule.is_multi_label=True \
datamodule.dataset=du_tc_multi \
datamodule.batch_size=24 \
datamodule.max_length=302 \
datamodule.pretrained_model=chinese-roberta-wwm-ext \
model=bert_text_multi_classification \
model.dropout=0.2 \
model.lr=3e-5 \
model.hidden_size=256 \
model.weight_decay=0.001 \
seed=11

