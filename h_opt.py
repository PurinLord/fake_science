from hyperopt import fmin, hp, tpe, Trials, STATUS_OK, STATUS_FAIL

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from fastai.text import * 


def objective(parms):
    class fbeta_binary(Callback):
        "Computes the f_beta between preds and targets for binary text classification"
    
        def __init__(self, beta2 = 1, eps=1e-9,sigmoid = True):      
            self.beta2=beta2**2
            self.eps = eps
            self.sigmoid = sigmoid
        
        def on_epoch_begin(self, **kwargs):
            self.TP = 0
            self.total_y_pred = 0   
            self.total_y_true = 0
        
        def on_batch_end(self, last_output, last_target, **kwargs):
            y_pred = last_output
            y_pred = y_pred.softmax(dim = 1)        
            y_pred = y_pred.argmax(dim=1)
            y_true = last_target.float()
            
            self.TP += ((y_pred==1) * (y_true==1)).float().sum()
            self.total_y_pred += (y_pred==1).float().sum()
            self.total_y_true += (y_true==1).float().sum()
        
        def on_epoch_end(self, **kwargs):
            prec = self.TP/(self.total_y_pred+self.eps)
            rec = self.TP/(self.total_y_true+self.eps)
            res = (prec*rec)/(prec*self.beta2+rec+self.eps)*(1+self.beta2)        
            #self.metric = res.mean()
            self.metric = res     


    try:
        data_clas = TextClasDataBunch.from_df( path, x_train, x_test, vocab=data_lm.train_ds.vocab, bs=int(parms['batch_size']))


        learn = language_model_learner(
                data_lm, pretrained_model=URLs.WT103,
                drop_mult=parms['drop_mult'],
                bptt=int(parms['bptt']),
                #emb_sz=int(parms['emb_sz']),
                #nh=int(parms['nh']),
                #nl=int(parms['nl']),
                pad_token=int(parms['pad_token']),
                tie_weights=parms['tie_weights'])
                #bias=parms['bias'],
                #qrnn=parms['qrnn'])

        learn.fit_one_cycle(1, 1e-2)

        learn.unfreeze()
        learn.fit_one_cycle(1, 1e-3)

        learn.save_encoder('ft_enc')

        learn = text_classifier_learner(
                data_clas,
                drop_mult=parms['c_drop_mult'],
                bptt=int(parms['c_bptt']),
                #emb_sz=int(parms['c_emb_sz']),
                #nh=int(parms['c_nh']),
                #nl=int(parms['c_nl']),
                pad_token=int(parms['c_pad_token']),
                #qrnn=parms['c_qrnn'],
                max_len=int(parms['max_len']),
                lin_ftrs=None,
                ps=None)
        #fbeta_binary = fbeta_binary()  # default is F1
        #learn.metrics.append(fbeta_binary) 

        learn.load_encoder('ft_enc')

        learn.fit_one_cycle(1, 1e-2)

        learn.freeze_to(-2)
        learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

        learn.unfreeze()
        learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
        
        valid = learn.validate()

        return {'loss': float(valid[2])*(-1), 'status': STATUS_OK}

    except Exception as e:
        print(e, {
            'batch_size': parms['batch_size'],
            'drop_mult':parms['drop_mult'],
            'bptt': parms['bptt'],
            'pad_token': parms['pad_token'],
            'tie_weights': parms['tie_weights'],
            #'qrnn': parms['qrnn'],

            'c_batch_size': parms['c_batch_size'],
            'c_drop_mult': parms['c_drop_mult'],
            'c_bptt': parms['c_bptt'],
            'c_pad_token': parms['c_pad_token'],
            #'c_qrnn': parms['c_qrnn'],
            'max_len': parms['max_len']
            })
        return {'loss': 100, 'status': STATUS_FAIL}


space = {
        'batch_size': hp.quniform('batch_size', 4, 32, q=1),
        'drop_mult': hp.uniform('drop_mult', 0.1, 0.9),
        'bptt': hp.quniform('bptt', 30, 110, q=1),
        #'emb_sz': hp.quniform('emb_sz', 10, 1000, q=10),
        #'nh': hp.quniform('nh', 100, 2500, q=10),
        #'nl': hp.quniform('nl', 1, 20, q=1),
        'pad_token': hp.quniform('pad_token', 1, 5, q=1),
        'tie_weights': hp.choice('tie_weights', [True, False]),
        #'bias': hp.choice('bias', [True, False]),
        #'qrnn': hp.choice('qrnn', [True, False]),

        'c_batch_size': hp.quniform('c_batch_size', 4, 32, q=1),
        'c_drop_mult': hp.uniform('c_drop_mult', 0.1, 0.9),
        'c_bptt': hp.quniform('c_bptt', 30, 110, q=1),
        #'c_emb_sz': hp.quniform('c_emb_sz', 10, 1000, q=10),
        #'c_nh': hp.quniform('c_nh', 100, 2500, q=10),
        #'c_nl': hp.quniform('c_nl', 1, 20, q=1),
        'c_pad_token': hp.quniform('c_pad_token', 1, 5, q=1),
        'c_qrnn': hp.choice('c_qrnn', [True, False]),
        'max_len': hp.quniform('max_len', 100, 2000, q=10)
        }


path = 'lm_data'

train_df = pandas.read_csv('data/cons_train_df.csv')
test_df = pandas.read_csv('data/cons_test_df.csv')

data_lm = TextLMDataBunch.from_df(path, x_train, x_test)

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100)

