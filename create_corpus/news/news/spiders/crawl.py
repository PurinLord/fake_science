import scrapy
import re
import os

from ES.indexes import News


class QuotesSpider(scrapy.Spider):
    name = "init"

    start_urls = ['https://mediabiasfactcheck.com/conspiracy/']

    def __init__(self):
        #self.start_url = 'https://yournewswire.com'
        self.visited = dict()


    def extract_site_name(self, current_site):
        site_name = re.search(
                'http.*//(?:www\.|)(.*)\..*', 
                current_site).group(1)
        return site_name

    
    def add_self_contained(self, response):
        local_url = response.xpath('//a[contains(@href, "%s")]/@href' % (response.meta['site_name'])).extract()
        relative_url = response.xpath('//a[starts-with(@href, "/")]/@href').extract()

        full_link = [
            response.meta['current_site'][:-1] + r 
            for r in relative_url
            ]

        local_url.extend(full_link)
        local_url = set(local_url)
        ## remove visited
        remaining_url = local_url - self.visited[response.meta['current_site']]

        response.meta['self_contained'] = remaining_url

        #missing = self.self_contained.difference(set(self.visited))


    def parse(self, response):
        r = response.css("a::attr(href)").extract()
        all_sites = r[43:309]
        #all_sites = r[48:98]; print('-- ALL --', all_sites)

        for url in all_sites:
            yield scrapy.Request(
                url,
                callback=self.get_site
            )



    def get_site(self, response):
        out_site = response.xpath('//p[contains(., "Source:")]/a/text()').extract_first()

        # extrae valor de conspiracy y pseudoscience
        # ['con1', 'pseudo5']
        cons_pseud = response.xpath('//img/@data-image-title').extract()

        # Factual reporting extract
        factual = response.xpath('//p[contains(., "Factual Reporting")]/span/strong/text()').extract_first()

        # Notes
        notes = response.xpath('//p[contains(., "Notes:")]/text()').extract_first()
        # Update
        update = response.xpath('//p[contains(., "UPDATE:")]/text()').extract_first()

        print('-- GET --', out_site)
        self.visited[out_site] = set()

        request = scrapy.Request(
                    out_site,
                    callback=self.crawl_site
                )
        request.meta['current_site'] = out_site
        request.meta['site_name'] = self.extract_site_name(out_site)
        request.meta['cons'] = int(re.findall('\d+', cons_pseud[0])[0])
        request.meta['preudo'] = int(re.findall('\d+', cons_pseud[1])[0])
        request.meta['factual'] = factual
        request.meta['notes'] = notes
        request.meta['update'] = update
        yield request


    def crawl_site(self, response):
        self.add_self_contained(response)

        for url in response.meta['self_contained']:
            print('-- CRAWL --', url)
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
        print('-- NEWS --')

        n = News()
        n.source = response.meta['current_site'] 
        n.url =  response.meta['site_name']
        n.content = news
        n.cons = response.meta['cons']
        n.preudo = response.meta['preudo']
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


    def clean_text(self, text):
        return  [e.strip() for e in text if self.is_valid(e)]


    def is_valid(self, text):
        text = text.strip()
        if text == '': return False
        if text[-1] not in {'.', '?', '!'}: return False

        if '{' in text: return False
        if '<' in text: return False
        return True
