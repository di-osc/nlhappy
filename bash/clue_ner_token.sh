#------------roberta-softmax best val/f1=0.757 -----------#
# python run.py \
# mode=exp \
# name=clue_ner_chartoken \
# trainer=ddp \
# trainer.max_epochs=20 \
# datamodule=token_classification \
# datamodule.pretrained_model=chinese-roberta-wwm-ext \
# datamodule.batch_size=64 \
# model=bert_softmax \
# model.hidden_size=256 \
# model.lr=5e-5 \
# model.dropout=0.2 \
# seed=3 \

#------------roberta-crf best val/f1=0.804 -----------#
# python run.py \
# mode=exp \
# name=clue_ner_chartoken \
# datamodule=token_classification \
# datamodule.dataset=clue_ner_chartoken \
# datamodule.pretrained_model=chinese-roberta-wwm-ext \
# model=bert_crf  \
# model.lr=3e-5 \
# model.dropout=0.3 \
# datamodule.batch_size=32 \
# trainer=ddp \
# seed=4


#-------------roberta-large best val/f1=0.81---------------------#
python run.py \
mode=exp \
name=clue_ner_chartoken \
trainer=ddp \
trainer.max_epochs=20 \
datamodule=token_classification \
datamodule.dataset=clue_ner_chartoken \
datamodule.pretrained_model=chinese-roberta-wwm-ext-large \
datamodule.batch_size=32 \
model=bert_crf \
model.lr=5e-5 \
model.dropout=0.3 \
seed=1 \




#------------- roberta-softmax train/f1=0.769 val/f1=0.744--------#
# python run.py \
# mode=exp \
# name=clue_ner \
# datamodule=token_classification \
# datamodule.tokenizer_name=hfl/chinese-roberta-wwm-ext \
# model=bert_softmax  \
# model.bert_name=hfl/chinese-roberta-wwm-ext \
# datamodule.batch_size=64 \
# trainer=ddp \
# trainer.max_epochs=50 \
# test_after_training=False \
# seed=22 \
# model.lr=5e-5 

#------------val/f1 = 0.739--------#
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





