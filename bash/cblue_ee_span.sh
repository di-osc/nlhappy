# ----------roberta val/best_f1 = 0.66----------------#
python run.py \
mode=exp \
name=cblue_ee_charspan \
trainer=ddp \
trainer.max_epochs=20 \
trainer.precision=16 \
test_after_training=False \
datamodule=span_classification \
datamodule.dataset=cblue_ee_charspan \
datamodule.batch_size=12 \
datamodule.max_length=420 \
datamodule.pretrained_model=chinese-roberta-wwm-ext \
model=bert_global_pointer \
model.dropout=0.2 \
model.lr=3e-5 \
model.hidden_size=256 \
model.weight_decay=0.0 \
seed=22


# ----------roberta-large val/best_f1 = 0.647----------------  #
# python run.py \
# mode=exp \
# name=cblue_ee_charspan \
# trainer=ddp \
# trainer.max_epochs=20 \
# trainer.precision=16 \
# test_after_training=False \
# datamodule=span_classification \
# datamodule.dataset=cblue_ee_charspan \
# datamodule.batch_size=2 \
# datamodule.max_length=415 \
# datamodule.pretrained_model=chinese-roberta-wwm-ext-large \
# model=bert_global_pointer \
# model.dropout=0.5 \
# model.lr=5e-5 \
# model.hidden_size=256 \
# model.weight_decay=0.0 \
# seed=22



