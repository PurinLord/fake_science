import numpy as np
import itertools

import pandas as pd
from create_corpus.news.ES.indexes import News
from elasticsearch_dsl.connections import connections

ELASTICSEARCH = connections.configure(
    default={
        'hosts': 'localhost:9200',
        'timeout': 60
    },
    sniff_on_start=True
)


def make_dataset(min_len=500, max_len=10000, cons_pseu='cons'):
    gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).scan()

    x, y = generate_dataset(gen, cons_pseu)
    
    size = len(y)
    X = np.zeros(size, dtype='<U'+str(max_len))
    Y = np.zeros(size, dtype='int')
    X = x
    Y = y

    return X, Y


def generate_dataset(gen, label_key, trim=0, encode=False):
    x = list()
    y = list()
    for n in gen:
        content = n.content
        if trim > 0:
            content = trim_text(content, trim)
        if encode:
            content = content.encode(encode, 'ignore')
        x.append(content)
        y.append(n[label_key])
    return np.array(x), np.array(y)


def trim_text(text, trim):
    split = text.split('\n')
    return ' '.join(split[trim:-trim])

def get_urls(search, num=1000, min_len= 10):
    agg_name = 'purin'
    search.aggs.bucket(agg_name, 'terms', field='source.keyword', size=num)
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


def smart_subset_select(urls, min_per, max_per, seed):
    total = sum([el['size'] for el in urls])
    min_total = int(total * min_per)
    max_total = int(total * max_per)
    gen = subset_sum(urls, min_total, max_total)
    while seed >= 0:
        subset = next(gen)
        seed -= 1
    train = [n['url'] for n in subset]
    all_names = [n['url'] for n in urls]
    test = list(set(all_names) - set(train))
    return train, test


def train_test_split(min_per, max_per, labels=range(1,6), seed=0, min_len=500, max_len=10000, trim = 3, encode=None, cons_pseu='cons'):

    train_sites = list()
    test_sites = list()

    for i in labels:

        print(cons_pseu, i)

        search = News().search().filter('range', length={'gte': min_len, 'lte': max_len})
        if cons_pseu == 'cons':
            search = search.filter('term', cons=i)
        elif cons_pseu == 'pseudo':
            search = search.filter('term', pseudo=i)

        urls = get_urls(search)

        #train, test = naiv_tari_test_clan_split(urls, min_per, max_per)
        train, test = smart_subset_select(urls, min_per, max_per, seed)

        train_sites.extend(train)
        test_sites.extend(test)


    train_gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).filter('terms', url__keyword=train_sites).scan()
    test_gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).filter('terms', url__keyword=test_sites).scan()

    x_train, y_train = generate_dataset(train_gen, cons_pseu, trim, encode)
    x_test, y_test = generate_dataset(test_gen, cons_pseu, trim, encode)

    return x_train, y_train, x_test, y_test


# https://stackoverflow.com/questions/4632322/finding-all-possible-combinations-of-numbers-to-reach-a-given-sum
def subset_sum(numbers, target_min, target_max, partial=[], partial_sum=0):
    if target_min <= partial_sum <= target_max:
        yield partial
    if partial_sum > target_max:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target_min, target_max, partial + [n], partial_sum + n['size'])


def make_df(x, y, x_name='x', y_name='y'):
    from pandas import DataFrame
    d = {y_name: list(y), x_name: list(x)}
    df = DataFrame(data=d)
    return df


def standarize(x_train, y_train, x_test, y_test, labels, save_path):
    col_names = ['labels', 'text']

    trn_idx = np.random.permutation(len(x_train))
    val_idx = np.random.permutation(len(x_test))

    trn_texts = x_train[trn_idx]
    val_texts = x_test[val_idx]

    trn_labels = y_train[trn_idx]
    val_labels = y_test[val_idx]

    df_trn = pd.DataFrame(
            {'text': trn_texts,
                'labels': trn_labels},
            columns=col_names)
    df_val = pd.DataFrame(
            {'text': val_texts,
                'labels': val_labels},
            columns=col_names)

    import pdb; pdb.set_trace()

    df_trn.to_csv(save_path + '/train.csv', header=False, index=False)
    df_val.to_csv(save_path + '/test.csv', header=False, index=False)

    #(save_path/'classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in labels)

