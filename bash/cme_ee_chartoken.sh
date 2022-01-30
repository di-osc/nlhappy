#-------------roberta-crf val/best_f1 0.666---------------------#
python run.py \
mode=exp \
trainer=ddp \
trainer.max_epochs=20 \
trainer.precision=16 \
datamodule=token_classification \
datamodule.dataset=cme_ee_chartoken \
datamodule.pretrained_model=chinese-roberta-wwm-ext \
datamodule.batch_size=16 \
datamodule.max_length=420 \
model=bert_crf \
model.lr=5e-5 \
model.dropout=0.3 \
seed=1 \