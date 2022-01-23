#----------bset val/f1 = 0.789----------------#
# python run.py \
# mode=exp \
# name=clue_ner_charspan \
# trainer=ddp \
# trainer.max_epochs=20 \
# trainer.precision=16 \
# test_after_training=False \
# datamodule=span_classification \
# datamodule.data_name=clue_ner_charspan \
# datamodule.batch_size=32 \
# datamodule.tokenizer_path=chinese-roberta-wwm-ext \
# model=global_pointer \
# model.bert_path=chinese-roberta-wwm-ext \
# model.dropout=0.2 \
# model.lr=3e-5 \
# seed=22


# ----------roberta val/best_f1 = 0.798----------------#
python run.py \
mode=exp \
name=clue_ner_charspan \
trainer=ddp \
trainer.max_epochs=20 \
trainer.precision=16 \
test_after_training=False \
datamodule=span_classification \
datamodule.dataset=clue_ner_charspan \
datamodule.batch_size=32 \
datamodule.pretrained_model=chinese-roberta-wwm-ext \
model=bert_global_pointer \
model.dropout=0.2 \
model.lr=3e-5 \
model.hidden_size=256 \
model.weight_decay=0.0 \
seed=22
