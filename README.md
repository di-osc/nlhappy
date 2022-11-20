
<div align='center'>

# NLHappy
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://spacy.io/"><img alt="Spacy" src="https://img.shields.io/badge/component-%20Spacy-blue"></a>
<a href="https://wandb.ai/"><img alt="WanDB" src="https://img.shields.io/badge/Log-WanDB-brightgreen"></a>
</div>
<br><br>

## 📌&nbsp;&nbsp; 简介

nlhappy致力于快速完成NLP任务,你唯一需要做的就是将数据处理为任务对应的数据类.
> 它主要的依赖有
- [transformers](https://huggingface.co/docs/transformers/index): 下载预训练权重
- [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/): 模型训练
- [datasets](https://huggingface.co/docs/datasets/index): 构建数据集
- [pydantic](https://wandb.ai/): 构建数据模型


## 🚀&nbsp;&nbsp; 安装
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
- 注册: https://wandb.ai/
- 获取认证: https://wandb.ai/authorize
- 登陆:
```bash
wandb login
```
模型训练开始后去[官网](https://wandb.ai/)查看训练实况
</details>




## ⚡&nbsp;&nbsp; 开始任务

<details>
<summary><b>文本分类</b></summary>

> 数据处理
```python
from nlhappy.utils.make_doc import Doc, DocBin
from nlhappy.utils.make_dataset import DatasetDict
# 构建corpus
# 将数据处理为统一的Doc对象,它存储着所有标签数据
docs = []
# data为你自己的数据
# doc._.label 为文本的标签,之所以加'_'是因为这是spacy Doc保存用户自己数据的用法
for d in data:
    doc = nlp(d['text'])
    doc._.label = d['label']
    docs.append(doc)
# 保存corpus,方便后边badcase分析
db = DocBin(docs=docs, store_user_data=True)
# 新闻文本-Tag3为保存格式目录,需要更换为自己的形式
db.to_disk('corpus/TNEWS-Tag15/train.spacy')
# 构建数据集,为了训练模型
ds = convert_docs_to_tc_dataset(docs=docs)
# 你可以将数据集转换为dataframe进行各种分析,比如获取文本最大长度
df = ds.to_pandas()
max_length = df['text'].str.len().max()
# 数据集切分
dsd = train_val_split(ds, val_frac=0.2)
# 保存数据集,注意要保存到datasets/目录下
dsd.save_to_disk('datasets/TNEWS')
```
> 训练模型

- 编写训练脚本,scripts/train.sh
```
nlhappy \
datamodule=text_classification \
datamodule.dataset=TNEWS \
datamodule.plm=hfl/chinese-roberta-wwm-ext \
datamodule.batch_size=32 \
model=bert_tc \
model.lr=3e-5 \
seed=1234
# 默认为单gpu 0号显卡训练,可以通过以下方式修改显卡
# trainer.devices=[1]
# 单卡半精度训练
# trainer.precision=16
# 使用wandb记录日志
# logger=wandb
# 多卡训练
# trainer=ddp trainer.devices=4
```

- 后台训练
```
nohup bash scripts/train.sh >/dev/null 2>&1 &
```
- 如果设置logger=wandb则现在可以去[wandb官网](https://wandb.ai/)查看训练详情了, 并且会自动产生logs目录里面包含了训练的ckpt,日志等信息.

> 构建自然语言处理流程,并添加组件
```python
import nlhappy

nlp = nlhappy.nlp()
# 默认device cpu, 阈值0.8
config = {'device':'cuda:0', 'threshold':0.9}
tc = nlp.add_pipe('text_classifier', config=config)
# logs文件夹里面训练的模型路径
ckpt = 'logs/experiments/runs/TNEWS/date/checkpoints/epoch_score.ckpt/'
tc.init_model(ckpt)
text = '文本'
doc = nlp(text)
# 查看结果
print(doc.text, doc._.label, doc.cats)
# 保存整个流程
nlp.to_disk('path/nlp')
# 加载
nlp = nlhappy.load('path/nlp')
```
> badcase分析
```python
import nlhappy
from nlhappy.utils.make_doc import get_docs_form_docbin
from nlhappy.utils.analysis_doc import analysis_text_badcase, Example

targs = get_docs_from_docbin('corpus/TNEWS-Tag15/train.spacy')
nlp = nlhappy.load('path/nlp')
preds = []
for d in targs:
    doc = nlp(d['text'])
    preds.append(doc)
eg = [Example(x,y) for x,y in zip(preds, targs)]
badcases, score = analysis_text_badcase(eg, return_prf=True)
print(badcases[0].x, badcases[0].x._.label)
print(badcases[0].y, badcases[0].y._.label)
```
> 部署
- 直接用nlp开发接口部署
- 转为onnx
```python
from nlhappy.models import BertTextClassification
ckpt = 'logs/path/ckpt'
model = BertTextClassification.load_from_ckeckpoint(ckpt)
model.to_onnx('path/tc.onnx')
model.tokenizer.save_pretrained('path/tokenizer')
```
</details>

<details>
<summary><b>实体抽取</b></summary>

nlhappy支持嵌套和非嵌套实体抽取任务
> 数据处理
```python
from nlhappy.utils.convert_doc import convert_spans_to_dataset
from nlhappy.utils.make_doc import get_docs_from_docbin
from nlhappy.utils.make_dataset import train_val_split
import nlhappy
# 制作docs
nlp = nlhappy.nlp()
docs = []
# data为你自己格式的原始数据,按需修改
# 只需设置doc.ents 
# 嵌套型实体设置doc.spans['all']
for d in data:
    doc = nlp(d['text'])
    # 非嵌套实体
    ents = []
    for ent in d['spans']:
        start = ent['start']
        end = ent['end']
        label = ent['label']
        span = doc.char_span(start, end, label)
        ents.append(span)
    doc.set_ents(ents)
    docs.append(doc)
    # 嵌套型实体
    for ent in d['spans']:
        start = ent['start']
        end = ent['end']
        label = ent['label']
        span = doc.char_span(start, end, label)
        doc.spans['all'].append(span)
    docs.append(doc)
# 保存docs,方便后边badcase分析
db = DocBin(docs=docs, store_user_data=True)
# 制作数据集
# 如果文本过长可以设置句子级别数据集
ds = convert_spans_to_dataset(docs, sentence_level=False)
dsd = train_val_split(ds, val_frac=0.2)
# 可以转换为dataframe分析数据
df = dsd.to_pandas()
max_length = df['text'].str.len().max()
# 保存数据集,注意要保存到datasets/目录下
dsd.save_to_disk('datasets/your_dataset_name')
```
> 训练模型
编写训练脚本
- 单卡
```bash
nlhappy \
datamodule=span_classification \
datamodule.dataset=your_dataset_name \
datamodule.max_length=2000 \
datamodule.batch_size=2 \
datamodule.plm=roberta-wwm-base \
model=global_pointer \
model.lr=3e-5 \
seed=22222
```
- 多卡
```
nlhappy \
trainer=ddp \
datamodule=span_classification \
datamodule.dataset=dataset_name \
datamodule.max_length=350 \
datamodule.batch_size=2 \
datamodule.plm=roberta-wwm-base \
model=global_pointer \
model.lr=3e-5 \
seed=22222
```
- 后台训练
```
nohup bash scripts/train.sh >/dev/null 2>&1 &
```
- 现在可以去[wandb官网](https://wandb.ai/)查看训练详情了, 并且会自动产生logs目录里面包含了训练的ckpt,日志等信息.
> 构建自然语言处理流程,并添加组件
```python
import nlhappy

nlp = nlhappy.nlp()
# 默认device cpu, 阈值0.8
config = {'device':'cuda:0', 'threshold':0.9, 'set_ents':True}
tc = nlp.add_pipe('span_classifier', config=config)
# logs文件夹里面训练的模型路径
ckpt = 'logs/experiments/runs/your_best_ckpt_path'
tc.init_model(ckpt)
text = '文本'
doc = nlp(text)
# 查看结果
# doc.ents 为非嵌套实体,如果有嵌套会选最大跨度实体
# doc.spans['all'] 可以包含嵌套实体
print(doc.text, doc.ents, doc.spans['all'])
# 保存整个流程
nlp.to_disk('path/nlp')
# 加载
nlp = nlhappy.load('path/nlp')
```
> badcase分析
```python
import nlhappy
from nlhappy.utils.analysis_doc import analysis_ent_badcase, Example, analysis_span_badcase
from nlhappy.utils.make_doc import get_docs_from_docbin

targs = get_docs_from_docbin('corpus/dataset_name/train.spacy')
nlp = nlhappy.load('path/nlp')
preds = []
for d in targs:
    doc = nlp(d['text'])
    preds.append(doc)
eg = [Example(x,y) for x,y in zip(preds, targs)]
# 非嵌套实体
badcases, score = analysis_ent_badcase(eg, return_prf=True)
print(badcases[0].x, badcases[0].x.ents)
print(badcases[0].y, badcases[0].y.ents)
# 嵌套实体
badcases, score = analysis_span_badcase(eg, return_prf=True)
print(badcases[0].x, badcases[0].x.spans['all'])
print(badcases[0].y, badcases[0].y.spans['all'])
```
> 部署
- 直接用nlp开发接口部署
- 转为onnx
```python
from nlhappy.models import GlobalPointer
ckpt = 'logs/path/ckpt'
model = GlobalPointer.load_from_ckeckpoint(ckpt)
model.to_onnx('path/tc.onnx')
model.tokenizer.save_pretrained('path/tokenizer')
```
</details>

<details>
<summary><b>实体标准化</b></summary>
TODO
</details>

<details>
<summary><b>关系抽取</b></summary>
TODO
</details>

<details>
<summary><b>事件抽取</b></summary>
TODO
</details>

<details>
<summary><b>通用信息抽取</b></summary>
TODO
</details>

<details>
<summary><b>摘要</b></summary>
TODO
</details>

<details>
<summary><b>翻译</b></summary>
TODO
</details>