#-------------对比roberta和bert的结果#

#------------- best val/f1 = 0.744--------#
python run.py \
mode=exp \
name=clue_ner \
datamodule=token_classification \
datamodule.tokenizer_name=hfl/chinese-roberta-wwm-ext \
model=bert_softmax  \
model.bert_name=hfl/chinese-roberta-wwm-ext \
datamodule.batch_size=64 \
trainer=ddp \
trainer.max_epochs=50 \
test_after_training=False \
seed=22 \
model.lr=5e-5 

#------------best val/f1 = 0.739--------#
# python run.py \
# mode=exp \
# name=clue_ner \
# datamodule=token_classification \
# datamodule.tokenizer_name=bert-base-chinese \
# model=bert_softmax  \
# model.bert_name=bert-base-chinese \
# datamodule.batch_size=64 \
# trainer=ddp \
# trainer.max_epochs=50 \
# test_after_training=False \
# seed=22 \
# model.lr=5e-5 \

# ------------roberta large best val/f1 = 0.751---------------#
# python run.py \
# mode=exp \
# name=clue_ner \
# datamodule=token_classification \
# datamodule.tokenizer_name=hfl/chinese-roberta-wwm-ext-large \
# model=bert_softmax  \
# model.bert_name=hfl/chinese-roberta-wwm-ext-large \
# model.lr=3e-5 \
# datamodule.batch_size=24 \
# trainer=ddp \
# trainer.max_epochs=50 \
# trainer.gpus=2 \
# test_after_training=False \
# seed=222





