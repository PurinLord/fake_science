import numpy as np
from collections import defaultdict
from random import shuffle
import copy
import csv

import pandas as pd
from create_corpus.news.ES.indexes import News#Test as News
from elasticsearch_dsl.connections import connections

connections.create_connection(hosts='localhost:9200')


def site_count(labels=range(1,6), min_len=500, max_len=10000, min_doc_count=10, label='cons'):
    out_dict = {"data": dict(),
            "charac": {
                "min_len": min_len,
                "max_len": max_len,
                "min_doc_count": min_doc_count,
                "label": label,
                "labels": labels
                }}

    for i in labels:

        print(label, i)

        search = News().search().filter('range', length={'gte': min_len, 'lte': max_len})
        if label == 'cons':
            search = search.filter('term', cons=i)
        elif label == 'pseudo':
            search = search.filter('term', pseudo=i)
        elif label == "factual":
            search = search.filter('term', factual=i)
        elif label == "bias":
            search = search.filter('term', bias=i)
        elif label == "science":
            if i == "con-science":
                search = search.filter('bool', must_not=[
                    {"bool": {"must": [
                        {"term": {"pseudo": {"value": 1}}},
                        {"term": {"cons": {"value": 1}}}
                        ]}}])
            if i == "pro-science":
                search = search.filter('bool', must=[
                    {"term": {"pseudo": {"value": 1}}},
                    {"term": {"cons": {"value": 1}}}
                    ])

        urls = get_urls(search, min_doc_count=min_doc_count)

        out_dict["data"][i] = urls

    return out_dict


def cross_validaton_split(site_count_dict, split_size=5, site_threshold=1):
    site_data = site_count_dict["data"]
    min_len = site_count_dict["charac"]["min_len"]
    max_len = site_count_dict["charac"]["max_len"]
    using_label = site_count_dict["charac"]["label"]

    split = defaultdict(dict)

    for label in site_data:
        urls = copy.deepcopy(site_data[label])
        total_docs = sum([el['size'] for el in urls])
        docs_per_part = total_docs//split_size
        sites_per_part = len(urls)/split_size
        for i in range(split_size-1):
            print(label, i)
            shuffle(urls)
            g = subset_sum(urls, docs_per_part, docs_per_part)
            doc_slice = next(g)
            while np.linalg.norm(len(doc_slice) - sites_per_part) > site_threshold:
                shuffle(urls)
                g = subset_sum(urls, docs_per_part, docs_per_part)
                doc_slice = next(g)
            for l in doc_slice:
                urls.pop(urls.index(l))
            split[label][i] = doc_slice
        split[label][split_size-1] = urls
    return split


def train_test_split(site_count_dict, min_per, max_per, seed=0, trim = 0, encode=None):
    site_data = site_count_dict["data"]
    min_len = site_count_dict["charac"]["min_len"]
    max_len = site_count_dict["charac"]["max_len"]
    using_label = site_count_dict["charac"]["label"]

    train_sites = list()
    test_sites = list()

    for label in site_data:

        urls = site_data[label]

        train, test = smart_subset_select(urls, min_per, max_per, seed)

        train_sites.extend(train)
        test_sites.extend(test)

    train_gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).filter('terms', source=train_sites).scan()
    test_gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).filter('terms', source=test_sites).scan()

    x_train, y_train = generate_dataset(train_gen, using_label, trim, encode)
    x_test, y_test = generate_dataset(test_gen, using_label, trim, encode)

    return x_train, y_train, x_test, y_test


def get_urls(search, num=1000, min_doc_count= 10):
    agg_name = 'purin'
    search.aggs.bucket(agg_name, 'terms', field='source', size=num)
    result = search.execute().to_dict()
    aggs = [
            {'url': a['key'], 'size': a['doc_count']} for a in 
            result['aggregations'][agg_name]['buckets']
            if a['doc_count'] >= min_doc_count
            ]
    return aggs


def generate_dataset(gen, label_key, trim=0, encode=False):
    x = list()
    y = list()
    for n in gen:
        content = n.content
        #content = trim_text(content, trim)
        if encode:
            content = content.encode(encode, 'ignore')
        x.append(content)
        y.append(n[label_key])
    return np.array(x), np.array(y)


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


# https://stackoverflow.com/questions/4632322/finding-all-possible-combinations-of-numbers-to-reach-a-given-sum
def subset_sum(numbers, target_min, target_max, partial=[], partial_sum=0):
    if target_min <= partial_sum <= target_max:
        yield partial
    if partial_sum > target_max:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target_min, target_max, partial + [n], partial_sum + n['size'])


def split_gens(split, min_len, max_len):
    num_splir = len(list(split.values())[0].keys())
    urls_split = defaultdict(list)
    for label in split:
        for s in split[label]:
            urls_split[s].extend([l["url"] for l in split[label][s]])

    gen_split = dict()
    for label in urls_split:
        gen_split[label] =  News().search().filter('range',
                length={'gte': min_len, 'lte': max_len}).filter('terms',
                        source=urls_split[label]).scan()
    return gen_split


def save_to_csv(gens, label , base_name):
    for split in gens:
        f = open("{}{}.csv".format(base_name, split), "w", newline="")
        csvwriter = csv.writer(f)
        for n in gens[split]:
            if label == "science":
                l = "pro-science" \
                        if n.cons == 1 and n.pseudo == 1 \
                        else "con-science"
            else:
                l = n.to_dict()[label].replace("\xa1", " ")
            csvwriter.writerow([
            l,
            n.content.replace("\n", " __new__linw__ ")])


def load_dfs(name):
    return [pd.read_csv("{}{}.csv".format(name, i), names=["label", "text"]) for i in range(5)]


def cros_valid_select(dfs, index):
    train = pd.concat([x for i,x in enumerate(dfs) if i!=index], sort=False)
    test = dfs[index]
    return train, test


def make_df(x, y, x_name='text', y_name='label'):
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


def trim_text(text, trim):
    print('trim not working')
    split = text.split('\n')
    if trim > 0:
        return ' '.join(split[trim:-trim])
    else:
        return ' '.join(split)

