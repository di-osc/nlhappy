#----------lcqmc--------------#

#----------cross-encoder--------------#
python  run.py \
datamodule=sentence_pair \
datamodule.num_workers=20 \
datamodule.batch_size=40 \
datamodule.pin_memory=True \
datamodule.sentence_max_length=50 \
model=bert_cross_encoder \
trainer=ddp \
logger=wandb \
logger.wandb.project=lcqmc \
logger.wandb.name=cross_encoder \

#----------bi-encoder--------------#
python run.py \
datamodule=sentence_pair \
datamodule.num_workers=20 \
datamodule.batch_size=64  \
datamodule.pin_memory=True \
datamodule.return_sentence_pair=True \
datamodule.sentence_max_length=50 \
model=bert_bi_encoder \
trainer=ddp \
trainer.gpus=4 \
trainer.max_epochs=20 \
logger=wandb \
logger.wandb.project=lcqmc \
logger.wandb.name=bi_encoder \
