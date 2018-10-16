import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder


from create_corpus.news.ES.indexes import News
from elasticsearch_dsl.connections import connections

ELASTICSEARCH = connections.configure(
    default={
        'hosts': 'localhost:9200',
        'timeout': 60
    },
    sniff_on_start=True
)


def make_label_enc(y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder


def generate_dataset(gen, label_key):
    x = list()
    y = list()
    for n in gen:
        x.append(n.content)
        y.append(n[label_key])
    return np.array(x), np.array(y)

def get_urls(search, num=1000, min_len= 10):
    agg_name = 'purin'
    search.aggs.bucket(agg_name, 'terms', field='url.keyword', size=num)
    result = search[0:0].execute().to_dict()
    aggs = [
            {'url': a['key'], 'size': a['doc_count']} for a in 
            result['aggregations'][agg_name]['buckets']
            if a['doc_count'] >= min_len
            ]
    return aggs

def naiv_tari_test_clan_split(urls, min_per, max_per):
    total = sum([el['size'] for el in urls])
    print('sources = ' + str(len(urls)))
    print('max = ' + str(urls[0]['size']) + ' '
          'min = ' + str(urls[-1]['size']))
    print('total = ' + str(total))
    for L in reversed(range(2, int(len(urls)/2))):
        for subset in itertools.combinations(urls, L):
            sub_total = sum([el['size'] for el in subset]) / total
            if min_per <= sub_total <= max_per:
                train = [n['url'] for n in subset]
                all_names = [n['url'] for n in urls]
                test = list(set(all_names) - set(train))
                return train, test
            if min_per <= (1 - sub_total) <= max_per:
                test = [n['url'] for n in subset]
                all_names = [n['url'] for n in urls]
                train = list(set(all_names) - set(test))
                return train, test


def make_dataset(min_len=500, max_len=10000):
    gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).scan()

    x, y = generate_dataset(gen, 'cons')
    
    size = len(y)
    X = np.zeros(size, dtype='<U'+str(max_len))
    Y = np.zeros(size, dtype='int')
    X = x
    Y = y

    return X, Y

def train_test_split(min_per, max_per):

    min_len = 500
    max_len = 10000
    
    train_sites = list()
    test_sites = list()
    
    for i in reversed(range(1, 6)):
    
        print('cons ', i)
    
        search = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).filter('term', cons=i)
    
        urls = get_urls(search)
    
        train, test = naiv_tari_test_clan_split(urls, 0.7, 0.75)
    
        train_sites.extend(train)
        test_sites.extend(test)
    
    
    train_gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).filter('terms', url__keyword=train_sites).scan()
    test_gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).filter('terms', url__keyword=test_sites).scan()
    
    x_train, y_train = generate_dataset(train_gen, 'cons')
    x_test, y_test = generate_dataset(test_gen, 'cons')

    return x_train, y_train, x_test, y_test
