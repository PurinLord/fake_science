import pickle

import numpy as np
from collections import Counter
from keras.activations import softmax
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, Lambda, Reshape, concatenate
from keras.layers.wrappers import Bidirectional
from keras.models import Input, Model
from keras.optimizers import Adamax
from keras.utils import to_categorical
from keras.utils import Sequence as KerasSeq
from sklearn.model_selection import train_test_split

from data_preprocessing import make_dataset, make_label_enc, TokenizerWrapper

ENC_WORDS = 1500
ENC_CHAR = 100
CUDA = True

if CUDA:
    from keras.layers import CuDNNLSTM


def make_input_encoder(max_words,
                       word_len,
                       individual_out,
                       rnn_drop=0.0,
                       cuda=False):
    if cuda:
        RNN = CuDNNLSTM
    else:
        RNN = LSTM

    input_embedding = Input(shape=(max_words, ))
    input_rnn = Input(shape=(
        max_words,
        word_len,
    ))

    embedding = Embedding(
        input_dim=ENC_WORDS, output_dim=individual_out)(input_embedding)

    rnnf = RNN(
        individual_out,
        return_sequences=True,
        return_state=False
        #dropout=rnn_drop\
    )(input_rnn)
    rnnb = RNN(
        individual_out,
        go_backwards=True,
        return_sequences=True,
        return_state=False
        #dropout=rnn_drop
    )(input_rnn)

    joint = concatenate([embedding, rnnf, rnnb])

    return Model(inputs=[input_embedding, input_rnn], outputs=joint)


def biRNN(num_memory_units, num_labels, input_encoder, cuda=False):
    if cuda:
        RNN = CuDNNLSTM
    else:
        RNN = LSTM

    bilstm = Bidirectional(RNN(num_memory_units))(input_encoder.output)
    mlp = Dense(num_labels, activation='tanh')(bilstm)
    output = Lambda(softmax)(mlp)

    return Model(inputs=input_encoder.input, outputs=output)


def make_model(max_words, in_len, cuda=False):
    model = make_input_encoder(max_words, in_len, 100, cuda=cuda)
    model = biRNN(32, 5, model, cuda=cuda)

    model.compile(
        optimizer=Adamax(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def make_dual_input(x, max_words, word_len):
    all_tokens = set()
    for el in x:
        all_tokens |= set(el.lower().split())
    input_encoder_word = TokenizerWrapper(
        nb_words=ENC_WORDS, max_input_length=max_words, char_level=False)
    input_encoder_lstm = TokenizerWrapper(
        nb_words=ENC_CHAR, max_input_length=word_len, char_level=True)
    input_encoder_word.fit(all_tokens)
    input_encoder_lstm.fit(all_tokens)
    return all_tokens, input_encoder_word, input_encoder_lstm


def encode_word(x, input_encoder_word):
    x_enc_word = np.zeros((len(x), input_encoder_word.max_input_length))
    for i in range(len(x)):
        x_enc_word[i] = input_encoder_word.transform([x[i]])
    return x_enc_word


def encode_char(x, input_encoder_lstm, max_words):
    x_enc_lstm = np.zeros((len(x), max_words,
                           input_encoder_lstm.max_input_length))
    for i in range(len(x)):
        split = x[i].split()
        limit = len(split)
        if limit > 10: limit = 10
        x_enc_lstm[i][:limit] = input_encoder_lstm.transform(split)[:limit]
    return x_enc_lstm


class Sequence(KerasSeq):
    def __init__(self, x_set, y_set, input_encoder_word, input_encoder_lstm,
                 label_encoder, batch_size):
        self.x = x_set
        self.y = y_set
        self.label_encoder = label_encoder
        self.input_encoder_word = input_encoder_word
        self.input_encoder_lstm = input_encoder_lstm
        self.batch_size = batch_size
        if self.y is not None:
            self.y = to_categorical(label_encoder.transform(self.y))



    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_w = encode_word(
            self.x[idx * self.batch_size:(idx + 1) * self.batch_size],
            self.input_encoder_word)
        batch_x_l = encode_char(
            self.x[idx * self.batch_size:(idx + 1) * self.batch_size],
            self.input_encoder_lstm, self.input_encoder_word.max_input_length)
        if self.y is not None:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_y = None

        return [batch_x_w, batch_x_l], batch_y


def eval_pred(start, end):
    x1 = x_enc_word[start:end]
    x2 = x_enc_lstm[start:end]
    pred = model.predict([x1, x2])
    inv_or = invert_label(y[start:end], label_encoder)
    inv_pred = invert_label(pred[start:end], label_encoder)
    return zip(inv_or, inv_pred, x[start:end])


if __name__ == '__main__':

    min_len = 500
    max_len = 10000
    max_words = 1500
    word_len = 25
    epochs = 10
    batch_size = 128


    x, y = make_dataset()

    label_encoder = make_label_enc(y)

    counter = dict(Counter(y))
    class_weight = {
        i: counter[name]
        for i, name in enumerate(label_encoder.classes_)
    }

    _, input_encoder_word, input_encoder_lstm = make_dual_input(
        x, max_words, word_len)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y)
    x_test, x_valid, y_test, y_valid = train_test_split(
        x_test, y_test, test_size=0.50, random_state=42, stratify=y_test)

    seq_train = Sequence(x_train, y_train, input_encoder_word,
                         input_encoder_lstm, label_encoder, batch_size)
    seq_valid = Sequence(x_valid, y_valid, input_encoder_word,
                         input_encoder_lstm, label_encoder, batch_size)
    seq_test = Sequence(x_test, y_test, input_encoder_word, input_encoder_lstm,
                        label_encoder, batch_size)

    model = make_model(max_words, word_len, CUDA)

    history = model.fit_generator(
        seq_train,
        epochs=epochs,
        validation_data=seq_valid,
        class_weight=class_weight,
        max_queue_size=10,
        use_multiprocessing=False,
        workers=1)

    print(model.evaluate_generator(seq_test))
