import scrapy
import re
import os


class QuotesSpider(scrapy.Spider):
    name = "init"

    start_urls = ['https://mediabiasfactcheck.com/conspiracy/']

    def __init__(self):
        #self.start_url = 'https://yournewswire.com'
        self.root = './extracted/'
        self.visited = dict()


    def create_filename(self, current_site):
        filename = re.search('http.*//(.*)\..*', current_site).group(1)
        filename += '.txt'
        return filename

    
    def add_self_contained(self, response):
        local_url = response.xpath('//a[starts-with(@href, "%s")]/@href' % (response.meta['current_site'])).extract()
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

        print('-- GET --', out_site)
        self.visited[out_site] = set()

        request = scrapy.Request(
                    out_site,
                    callback=self.crawl_site
                )
        request.meta['current_site'] = out_site 
        request.meta['filename'] = self.create_filename(out_site)
        # open file???
        yield request
        # Close file??


    def crawl_site(self, response):
        self.add_self_contained(response)

        for url in response.meta['self_contained']:
            print('-- CRAWL --', url)
            self.visited[response.meta['current_site']].add(url)
            request = scrapy.Request(
                url,
                callback=self.extract_news
            )
            request.meta['current_site'] = response.meta['current_site']
            request.meta['filename'] = response.meta['filename']
            yield request
    
    def extract_news(self, response):
        #self.log('%s' % response.url)
        # Buscar la alternancia de ligas contenidas y texto (imagenes)
        alternating = response.xpath('//a[starts-with(@href, "%s")]/following::text()' % (response.meta['current_site'])).extract()
        clean_alt = self.clean_text(alternating)

        text = response.xpath('//*/text()').extract()
        clean = self.clean_text(text)
        joined = '\n'.join(clean)
        news = response.url + '\n' + joined + '\n\n'
        print('-- NEWS --')
        #response.meta['file'].write(news)
        f = open(self.root+response.meta['filename'], 'a')
        f.write(news)
        f.close()
        
        # Recursive call
        request = scrapy.Request(
                    response.url,
                    callback=self.crawl_site
                )
        request.meta['current_site'] = response.meta['current_site']
        request.meta['filename'] = response.meta['filename']
        yield request


    def clean_text(self, text):
        return  [e.strip() for e in text if self.is_valid(e)]


    def is_valid(self, text):
        text = text.strip()
        if text == '': return False
        if text[-1] not in {'.', '?', '!'}: return False
        # ]
        if '{' in text: return False
        if '<' in text: return False
        return True
