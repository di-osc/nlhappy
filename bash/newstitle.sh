#-------------newstitle-----------#
#------roberta对比bert-base---------#
#bert-base-chinese
python run.py \
datamodule=text_classification \
datamodule.batch_size=96 \
datamodule.num_workers=40 \
datamodule.pin_memory=True \
model=bert_sequence_classification \
model.output_size=14 \ 
trainer=ddp \
logger=wandb \
logger.wandb.project=newstitle \
logger.wandb.name=bert-base-chinese \

#roberta-base
python run.py \
datamodule=text_classification \
datamodule.tokenizer_name=hfl/chinese-roberta-wwm-ext \
datamodule.batch_size=96 \
datamodule.num_workers=40 \
datamodule.pin_memory=True \
model=bert_sequence_classification \
model.bert_name=hfl/chinese-roberta-wwm-ext \
model.output_size=14 \
trainer=ddp \
logger=wandb \
logger.wandb.project=newstitle \
logger.wandb.name=chinese-roberta-wwm-ext \