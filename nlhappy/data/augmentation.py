import requests


def augment_text_from_youdao_translator(text: str):
    """通过调用有道翻译接口增加文本数据,最多可以扩充三条文本

    Args:
        text (str): 原始文本

    Returns:
        List[str]: 增强的文本,不包含原始文本. 
    """
    results = []
    trans_types = [('ZH_CN2EN','EN2ZH_CN'),('ZH_CN2KR','KR2ZH_CN'),('ZH_CN2JA','JA2ZH_CN'),]
    for t in trans_types:
        try:
            data1 = { 'doctype': 'json', 'type': t[0],'i': text }
            r = requests.get("http://fanyi.youdao.com/translate",params=data1)
            r = r.json()
            tgt = r['translateResult'][0][0]['tgt']
            
            data2 = { 'doctype': 'json', 'type': t[1],'i': tgt}
            r = requests.get("http://fanyi.youdao.com/translate",params=data2)
            r = r.json()
            res = r['translateResult'][0][0]['tgt']
            if res not in results and res!=text:
                results.append(res)
        except:
            pass
    return results  