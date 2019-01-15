#-e NVIDIA_VISIBLE_DEVICES=0
# PYTHONIOENCODING='utf8'

from fastai import *
from fastai.text import * 

from dataset_creation import train_test_split


x_train, y_train, x_test, y_test = train_test_split(0.8, 0.8, labels=[1,5], seed=4985)

d = {'label': y_train.astype(str), 'text':x_train}
train_df = DataFrame(data=d)
                                    
d = {'label': y_test.astype(str), 'text':x_test}
test_df = DataFrame(data=d)

path = 'lm_data'

data_lm = TextLMDataBunch.from_df(path, train_df, test_df)
data_clas = TextClasDataBunch.from_df(path, train_df, test_df, vocab=data_lm.train_ds.vocab, bs=32)

data_lm.save()
data_clas.save()

data_lm = TextLMDataBunch.load(path)
data_clas = TextClasDataBunch.load(path, bs=32)

learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)

learn.predict("I like the", n_words=10)

learn.save_encoder('ft_enc')

learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('ft_enc')

data_clas.show_batch()

learn.fit_one_cycle(1, 1e-2)

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))

learn.predict("My screen does not load")

learn.save('clas')

