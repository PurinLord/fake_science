import scrapy
import re
import csv

from ES.indexes import News


class QuotesSpider(scrapy.Spider):
    name = "factual"

    r = csv.reader(open("news_base.csv", "r"))
    next(r)
    source_data = {
                el[2]: {
                    "source_url": el[0],
                    "source_url_processed": el[1],
                    "URL": el[2],
                    "fact": el[3],
                    "bias": el[4]
                    }
                for el in r}
    start_urls = list(source_data.keys())

    def __init__(self):
        #self.start_url = 'https://yournewswire.com'
        self.visited = dict()

    def parse(self, response):
        out_site = response.url

        left_rigth = source_data[out_site]["bias"]

        # Factual reporting extract
        factual = source_data[out_site]["fact"]

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
        request.meta['site_name'] = source_data[out_site]["source_url_processed"]
        request.meta['left_rigth'] = left_rigth
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

        # Recursive call
        request = scrapy.Request(
                    response.url,
                    callback=self.crawl_site
                )
        request.meta.update(response.meta)
        yield request

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

        full_link = [
            response.meta['current_site'].strip('/') + r
            for r in relative_url
            ]

        local_url.extend(full_link)
        local_url = set(local_url)
        # remove visited
        remaining_url = local_url - self.visited[response.meta['current_site']]

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

