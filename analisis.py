import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split



from create_corpus.news.ES.indexes import News
from elasticsearch_dsl.connections import connections

ELASTICSEARCH = connections.configure(
    default={
        'hosts': 'localhost:9200',
        'timeout': 60
    },
    sniff_on_start=True
)


def generate_dataset(gen, label_key):
    x = list()
    y = list()
    for n in gen:
        x.append(n.content)
        y.append(n[label_key])
    return np.array(x), np.array(y)


min_len = 500
max_len = 10000

gen = News().search().filter('range', length={'gte': min_len, 'lte': max_len}).scan()

data, label = generate_dataset(gen, 'cons')

x_train, x_test, y_train, y_test = train_test_split(data, label, 
                                                    random_state=42)
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(x_train)
x = tfidf_vect.transform(x_train)

model = SVC(C=10000)
model.fit(x, y_train)

pred = model.predict(tfidf_vect.transform(x_test))

from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test, pred))
