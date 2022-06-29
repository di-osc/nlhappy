
explain_dict = {
    'PSE': 'Prompt Span Extraction, 基于提示的文本抽取简称',
    'TC': 'text classification, 文本分类简称',
    'EE': 'Event Extraction, 事件抽取简称',
    'ToC': 'Token Classification, 基于token的分类简称',
    'DuUIE': '2022CCKS通用信息抽取数据集, 包含实体抽取、关系抽取、事件抽取和情感抽取四个任务.',
    'TPC': 'text-pair classification, 文本对分类简称',
    'TPS': 'text-pair similarity 文本对相似度的简称',
    'ChnSentiCorp': '经典的句子级情感分类数据集，包含酒店、笔记本电脑和数据相关的网络评论数据，共包含积极、消极两个类别。',
    'JDNER': '京东商城商品标题命名实体识别数据集. 标签全部为脱敏数字, 标签非常不均衡.',
    'TNEWS': '今日头条中文新闻（短文本）分类, 该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等',
    'CMeIE': 'CBLUE排行榜中中文医学文本实体关系抽取任务数据集,文本中含有@,并且语义模糊.',
    'CMeEE': 'CBLUE排行榜中中文医学命名实体识别数据集, 含有嵌套的情况.',
    'LCQMC': '中文问答匹配数据集, 238766训练集、8802验证集和12500测试集',
    'ICD-STS': 'ICD-STS数据集, 包含训练集、验证集和测试集',
    'CHIP-CDN': 'CBLUE排行榜中临床术语标准化任务的数据集, 诊断词与标准词含有一对多的情况',
    'CHIP-CTC': '临床试验筛选标准短文本分类, https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#1',
    'CHIP-STS': '平安医疗科技疾病问答迁移学习, https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#1',
    'IMCS': '该数据集收集了真实的在线医患对话，并进行了多层次（Multi-Level）的人工标注，包含命名实体、对话意图、症状标签、医疗报告等，并在CCL 2021会议上举办了第一届智能对话诊疗评测比赛 （http://www.fudan-disc.com/sharedtask/imcs21/index.html）',
    'EMR_NER-PSE': '基于ccks2019医疗电子病历实体识别数据集制作的prompt-span-extraction数据集',
    'DuEE-Fin': '金融领域篇章级别事件抽取数据集， 共包含13个已定义好的事件类型约束和1.15万中文篇章（存在部分非目标篇章作为负样例），其中6900训练集，1150验证集和3450测试集，'
}



def explain(key: str):

    """
    Explain the meaning of a key.

    :param key: The key to explain.
    :return: The explanation.
    """

    if key in explain_dict:
        return explain_dict[key]
    else:
        return 'No explanation for this key.'
    
def chinese(tokenizer: str = 'char'):
    import spacy
    from nlhappy.components.tokenizer import all_tokenizers
    tokenizer = all_tokenizers[tokenizer]
    nlp = spacy.blank('zh')
    nlp.tokenizer = tokenizer(nlp.vocab)
    return nlp

def nlp():
    import spacy
    return spacy.blank('zh')


def load(path: str, disable: list=[]):
    """load spacy nlp pipeline from path

    Args:
        path (str): path to the pipeline
        disable(list): a list of pipe names to disable

    Returns:
        nlp: spacy nlp pipeline
    """
    import spacy
    nlp = spacy.load(name=path, disable=disable)
    return nlp

    
    
