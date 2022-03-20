
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

    
    else: return "No such key"