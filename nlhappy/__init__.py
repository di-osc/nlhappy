# from .components.tokenizer import CharTokenizer
# import spacy



# def make_zh_nlp(tokenize_type: str = 'char'):
#     '''构建适合中文的自然语言处理基础流程,只包含tokenizer'''
#     nlp = spacy.blank('zh')
#     if tokenize_type == 'char':
#         nlp.tokenizer = CharTokenizer(vocab=nlp.vocab)
#     return nlp



def explain(key: str):

    """
    Explain the meaning of a key.

    :param key: The key to explain.
    :return: The explanation.
    """

    if key == "ICD-STS":
        return "医疗术语语义相似度任务,在ICD10上根据章节,亚目,类目的不同筛选出的疾病名称句子对,相似度分别为0, 0.1, 0.3, 0.5"
    elif key == "CHIP-CDN":
        return "CBLUE排行榜中临床术语标准化任务的数据集, 诊断词与标准词含有一对多的情况"
    elif key == "CMeIE":
        return "CBLUE排行榜中中文医学文本实体关系抽取任务数据集,文本中含有@,并且语义模糊."
    elif key == "CMeEE":
        return "CBLUE排行榜中中文医学命名实体识别数据集, 含有嵌套的情况."
    elif key == "LCQMC":
        return "中文问答匹配数据集, 238766训练集、8802验证集和12500测试集"
    elif key == "CHIP-CTC":
        return "临床试验筛选标准短文本分类, https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#1"
    elif key == "CHIP-STS":
        return "平安医疗科技疾病问答迁移学习, https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#1"
    elif key == "TNEWS":
        return "今日头条中文新闻（短文本）分类, 该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等"
    elif key == "JDNER":
        return "京东商城商品标题命名实体识别数据集. 标签全部为脱敏数字, 标签非常不均衡."
    elif key == "ChnSentiCorp":
        return "经典的句子级情感分类数据集，包含酒店、笔记本电脑和数据相关的网络评论数据，共包含积极、消极两个类别。"
    elif key == "DuUIE":
        return "2022CCKS通用信息抽取数据集, 包含实体抽取、关系抽取、事件抽取和情感抽取四个任务."
    
    
    else: return "No such key"