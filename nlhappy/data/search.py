from lxml.etree import HTML
import requests
from pydantic import BaseModel, constr, HttpUrl
from typing import List
from urllib.parse import unquote
from unicodedata import normalize


class Item(BaseModel):
    """百科的一个条目
    name: 词条名称
    desc: 词条描述,用于混淆词条
    summary: 词条摘要
    synonym: 同义词
    """
    url: HttpUrl = ''
    name: constr(strip_whitespace=True)
    desc: constr(strip_whitespace=True)
    summary: constr(strip_whitespace=True)
    synonyms: List[constr(strip_whitespace=True, min_length=1)]


class BaiduPedia:
    """百度百科
    """
    item_base_url = "https://baike.baidu.com/item/"
    search_base_url = 'https://baike.baidu.com/search?word='
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36' }

    def _get_item_url(self, item_query: str) -> str:
        return self.item_base_url + item_query
    
    def _get_item_html(self, url: str) -> HTML:
        """根据地址读取静态页面
        """
        res = requests.get(url=url, headers=self.headers)
        res.encoding = 'utf-8'
        html = HTML(text=res.text)
        return html
        
    def _get_item_summary(self, html) -> str:
        """获取该词条的简介
        """
        sum_ls = html.xpath('//div[@class="lemma-summary"]//text()')
        summary = ''.join([item.strip('\n') for item in sum_ls])
        summary = normalize('NFKC', summary) # 将\xa0类似的unicode字符转换为正常字符
        return summary
    
    def _get_item_desc(self, html) -> str:
        """获取该词条的描述
        """
        desc_ls = html.xpath('//div[@class="lemma-desc"]//text()')
        return ''.join(desc_ls)
    
    def _get_item_name(self, html) -> str:
        """获取该词条中的名称
        """
        item_name: List = html.xpath('//h1/text()')
        assert len(item_name) == 1
        return item_name[0]
    
    def _get_item_synonyms(self, html) -> List[str]:
        """获取该词条的所有同义词
        """
        return html.xpath('//a[@title="同义词"]//following-sibling::span/text()')
    
    def _parse_item_html(self, html):
        item_name = self._get_item_name(html)
        item_summary = self._get_item_summary(html)
        item_desc = self._get_item_desc(html)
        item_synonyms = self._get_item_synonyms(html=html)
        return Item(name=item_name, summary=item_summary, desc=item_desc, synonyms=item_synonyms)
    
    def _get_search_url(self, search_text) -> str:
        return self.search_base_url + search_text
    
    def _get_search_html(self, url: str):
        res = requests.get(url=url, headers=self.headers)
        res.encoding = 'utf-8'
        html = HTML(text=res.text)
        return html
    
    def _has_search_results(self, html) -> bool:
        no_results = html.xpath('//div[@class="no-result"]/text()')
        return len(no_results) == 0
    
    def _get_searched_item_urls(self, html) -> List[str]:
        urls = []
        for a in html.xpath('//a[@class="result-title J-result-title"]'):
            item_query_ls: List = a.get('href').split('/item/')[-1].split('/')
            item_query = unquote(item_query_ls.pop(0)) # 将urlencode 转码为字符串
            item_query_ls  = [item_query] + item_query_ls
            item_query = '/'.join(item_query_ls)
            item_url = self._get_item_url(item_query=item_query)
            urls.append(item_url)
        return urls
    
    def search(self, search_text: str, num_return_items: int = 1) -> List[Item]:
        """根据搜索的内容自动搜索百科条目

        Args:
            search_text (str): 带搜索的文本
            num_return_items (int, optional): 返回的搜索的条目. Defaults to 1.

        Returns:
            List[Item]: 所有返回条目
        """
        items = []
        url = self._get_search_url(search_text=search_text)
        html = self._get_search_html(url=url)
        if self._has_search_results(html=html):
            urls = self._get_searched_item_urls(html=html)
            num_items = int(max(min(num_return_items, len(urls)), 1))
            for url in urls[0: num_items]:
                item_html = self._get_item_html(url)
                item = self._parse_item_html(html=item_html)
                item.url = url
                items.append(item)
        return items