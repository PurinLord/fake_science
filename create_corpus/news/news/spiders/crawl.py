import scrapy
import re
from urllib import parse
import elasticsearch

from ES.indexes import News


class QuotesSpider(scrapy.Spider):
    name = "init"

    #start_urls = ['https://mediabiasfactcheck.com/pro-science']
    start_urls = ['https://mediabiasfactcheck.com/conspiracy/']

    def __init__(self):
        #self.start_url = 'https://yournewswire.com'
        self.visited = dict()
        self.index_client = elasticsearch.client.IndicesClient(
                client=elasticsearch.Elasticsearch(hosts="database:9200"))


    def parse(self, response):
        r = response.css("a::attr(href)").extract()
        all_sites = r[41:329]
        #all_sites = r[48:98]; print('-- ALL --', all_sites)

        for url in all_sites:
            if "real-jew-news" in url:
                continue
            yield scrapy.Request(
                url,
                callback=self.get_site
            )

    def get_site(self, response):
        out_site = response.xpath(
                '//p[contains(., "Source:")]/a/text()').extract_first()

        # extrae valor de conspiracy y pseudoscience
        # ['con1', 'pseudo5']
        cons_pseud = response.xpath('//img/@data-image-title').extract()

        # Factual reporting extract
        factual = response.xpath(
                '//p[contains(., "Factual Reporting")]/span/strong/text()'
                ).extract_first()

        # Notes
        notes = response.xpath(
                '//p[contains(., "Notes:")]/text()').extract_first()
        # Update
        update = response.xpath(
                '//p[contains(., "UPDATE:")]/text()').extract_first()

        # print('-- GET --', out_site)
        self.visited[out_site] = set()

        request = scrapy.Request(
                    out_site,
                    callback=self.crawl_site
                )
        request.meta['current_site'] = out_site
        request.meta['site_name'] = self.extract_site_name(out_site)
        if cons_pseud != []:
            request.meta['cons'] = int(re.findall('\d+', cons_pseud[0])[0])
            request.meta['pseudo'] = int(re.findall('\d+', cons_pseud[1])[0])
        else:
            request.meta['cons'] = 1
            request.meta['pseudo'] = 1

        request.meta['factual'] = factual
        request.meta['notes'] = notes
        request.meta['update'] = update
        yield request

    def crawl_site(self, response):
        self.add_self_contained(response)

        for url in response.meta['self_contained']:
            # print('-- CRAWL --', url)
            self.visited[response.meta['current_site']].add(url)
            request = scrapy.Request(
                url,
                callback=self.extract_news
            )
            request.meta.update(response.meta)
            yield request

    def extract_news(self, response):

        text = response.xpath('//*/text()').extract()
        clean = self.clean_text(text)
        news = '\n'.join(clean)
        # print('-- NEWS --')

        self.save_news(news, response)

        # Recursive call
        request = scrapy.Request(
                    response.url,
                    callback=self.crawl_site
                )
        request.meta.update(response.meta)
        yield request

    def save_news(self, news, response):
        try:
            n = News()
            n.source = response.meta['current_site']
            n.url = response.url
            n.content = news
            n.length = len(news)
            n.cons = response.meta['cons']
            n.pseudo = response.meta['pseudo']
            n.factual = response.meta['factual']
            n.notes = response.meta['notes']
            n.update = response.meta['update']
            n.save()
        except:
            data = '{"index.blocks.read_only_allow_delete": null}'
            self.index_client.put_settings(index="news_3", body=data)
            self.save_news(news, response)

    def extract_site_name(self, current_site):
        site_name = re.search(
                'http.*//(?:www\.|)(.*)\..*', 
                current_site).group(1)
        return site_name

    def add_self_contained(self, response):
        local_url = response.xpath(
                '//a[contains(@href, "%s")]/@href' % (
                    response.meta['site_name'])).extract()

        relative_url = response.xpath(
                '//a[starts-with(@href, "/")]/@href').extract()
        if isinstance(relative_url, list):
            local_url.extend(relative_url)

        full_link = {
            parse.urljoin(response.meta['current_site'], r)
            for r in local_url
            }

        # remove visited
        remaining_url = full_link - self.visited[response.meta['current_site']]

        response.meta['self_contained'] = remaining_url

        #missing = self.self_contained.difference(set(self.visited))
    def clean_text(self, text):
        return [e.strip() for e in text if self.is_valid(e)]

    def is_valid(self, text):
        text = text.strip()
        if text == '': return False
        if text[-1] not in {'.', '?', '!'}: return False

        if '{' in text: return False
        if '<' in text: return False
        return True

