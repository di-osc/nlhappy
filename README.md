

# NLHAPPY

复现自然语言处理的模型

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://spacy.io/"><img alt="Spacy" src="https://img.shields.io/badge/component-%20Spacy-blue"></a>
<a href="https://wandb.ai/"><img alt="WanDB" src="https://img.shields.io/badge/Log-WanDB-brightgreen"></a>
<br>



</div>




## 安装并运行

安装依赖

```bash
# pip 安装
pip install -U pip
pip install -U nlhappy


# 通过poetry打包然后安装
# 首先将文件下载到本地
# 通过pipx 安装poetry
pip install -U pipx
pipx install poetry
pipx ensurepath 
# 需要重新打开命令行
poetry build
# 安装包 在dist文件夹

```

训练模型

```bash
# 单卡运行
nlhappy \
datamodule=text_classification \
datamodule.dataset=TNEWS \
datamodule.max_length=100 \
datamodule.plm=chinese-roberta-wwm-ext \
model=bert_text_classification \
model.lr=1e-5 \
model.hidden_size=256 \
model.dropout=0.1 \
model.weight_decay=0.01 \
seed=123

# 多卡运行
nlhappy \
trainer=ddp \
trainer.gpus=4 \
datamodule=prompt_span_extraction \
datamodule.dataset=CLUENER_PSE \
datamodule.plm=chinese-roberta-wwm-ext \
datamodule.max_length=100 \
datamodule.batch_size=16 \
model=bert_global_span \
seed=12345

# 快速调试
nlhappy \
trainer=debug \
datamodule=prompt_span_extraction \
datamodule.dataset=CLUENER_PSE \
datamodule.plm=chinese-roberta-wwm-ext \
datamodule.max_length=100 \
datamodule.batch_size=16 \
model=bert_global_span 

```

## 可用任务

- text_pair_classification(适用于单标签文本对分类任务)
- text_pair_regression (适用于文本相似度任务)
- text_classification (适用于单标签文本分类任务)
- span_classification (适用于嵌套型实体识别等任务)
- token_classification (适用于序列标注任务)
- triple_extraction (适用于三元组抽取)
- prompt_span_extraction (适用于通用信息抽取等任务)




