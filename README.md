
<div align='center'>

# nlhappy
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://wandb.ai/"><img alt="WanDB" src="https://img.shields.io/badge/Log-WanDB-brightgreen"></a>
</div>
<br><br>

## 📌&nbsp;&nbsp; 简介

nlhappy致力于复现自然语言处理各类任务的SOTA模型(不包含LLM相关任务)。

> 文档地址:
- [notion文档](https://wangmengdi.notion.site/NLHAPPY-264f05d1084848efa42068c83539904a)
> 它主要的依赖有
- [transformers](https://huggingface.co/docs/transformers/index): 下载预训练权重
- [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/): 模型训练
- [datasets](https://huggingface.co/docs/datasets/index): 构建数据集
- [pydantic](https://wandb.ai/): 数据校验

## 📌&nbsp;&nbsp; 支持NLP任务

- [x] 实体抽取
- [x] 嵌套实体抽取
- [x] 非连续实体抽取
- [x] 关系抽取
- [x] 事件抽取
- [x] 文本单标签分类
- [x] 文本多标签分类
- [x] 阅读理解
- [x] 文本对分类
- [x] 文本对相似度(文本向量化)


## 📌&nbsp;&nbsp; 安装
<details>
<summary><b>安装nlhappy</b></summary>

> 推荐先去[pytorch官网](https://pytorch.org/get-started/locally/)安装pytorch和对应cuda
```bash
# pip 安装
pip install --upgrade pip
pip install --upgrade nlhappy
```
</details>

<details>
<summary><b>其他可选</b></summary>

> 推荐安装wandb用于可视化训练日志
- 安装: 
```bash
pip install wandb 
```
- 注册: https://wandb.ai/
- 获取认证: https://wandb.ai/authorize
- 登陆:
```bash
wandb login
```
- 使用
```
# 命令行训练
nlhappy datamodule=xxx model=xxx trainer=xxx logger=wandb
```
模型训练开始后去[官网](https://wandb.ai/)查看训练实况
</details>