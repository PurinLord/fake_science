import scrapy
import re

class QuotesSpider(scrapy.Spider):
    name = "init"

    start_urls = ['https://mediabiasfactcheck.com/conspiracy/']

    def __init__(self):
        #self.start_url = 'https://yournewswire.com'
        self.current_site = ''
        self.root = './extracted/'
        self.filename = ''
        self.visited = set()
        self.self_contained = list()

        self.news = list()


    def set_filename(self):
        self.filename = re.search('http.*//(.*)\..*', self.current_site).group(1)
        self.filename += '.txt'


    
    def add_site_scope(self, response):
        r = response.css("a::attr(href)").extract()
        new_url = {url for url in r if url.startswith(self.current_site)}
        missing = new_url.difference(set(self.self_contained))
        self.self_contained.extend(missing)

        #missing = self.self_contained.difference(set(self.visited))


    def parse(self, response):
        r = response.css("a::attr(href)").extract()
        #all_sites = r[43:407]
        all_sites = r[100:102]; print('-- ALL --', all_sites)
        for url in all_sites:
            yield scrapy.Request(
                url,
                callback=self.get_site
            )



    def get_site(self, response):
        out_site = response.xpath('//p[contains(., "Source:")]/a/text()').extract_first()
        self.current_site = out_site
        self.set_filename()
        print('-- GET --', self.current_site)
        yield scrapy.Request(
                    out_site,
                    callback=self.crawl_site
                )


    def crawl_site(self, response):
        self.visited = set()
        self.self_contained = list()
        # lista de todas las ligas
        self.add_site_scope(response)

        for url in self.self_contained:
            print('-- CRAWL --', url)
            if url not in self.visited:
                self.visited.add(url)
                yield scrapy.Request(
                    url,
                    callback=self.extract_news
                )

    
    def extract_news(self, response):
        #self.log('%s' % response.url)
        self.add_site_scope(response)
        text = response.xpath('//*/text()').extract()
        clean = [e.strip() for e in text if self.is_valid(e)]
        joined = '\n'.join(clean)
        news = response.url + '\n' + joined + '\n\n'
        print('-- NEWS --', self.root+self.filename)
        with open(self.root+self.filename, 'a') as f:
            f.write(news)


    def is_valid(self, text):
        text = text.strip()
        if text == '': return False
        if text[-1] != '.': return False
        if '{' in text: return False
        if '<' in text: return False
        return True
