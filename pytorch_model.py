from fastai import *
from fastai.text import * 

'''
from dataset_creation import train_test_split, make_df


x_train, y_train, x_test, y_test = train_test_split(0.8, 0.8, labels=[1,5], seed=4985, encode=None)

train_df = make_df(x_train, y_train.astype(str), x_name='text', y_name='label')
                                    
test_df = make_df(x_test, y_test.astype(str), x_name='text', y_name='label')

# Language model data
data_lm = TextLMDataBunch.from_df('.', train_df, test_df)
'''

data_lm = TextLMDataBunch.load('lm_data', '')

# Classifier model data
#data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)

learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)

learn.predict("This is a review about", n_words=10)

#data_clas = TextClasDataBunch.from_df('.', train_df, test_df, vocab=data_lm.train_ds.vocab, bs=32)
#learn = text_classifier_learner(data_clas, drop_mult=0.5)
#learn.load_encoder('ft_enc')
#learn.fit_one_cycle(1, 1e-2)
#learn.freeze_to(-2)
#learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
#learn.unfreeze()
#learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
#learn.predict("Trump lies!")

