import re
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def make_label_enc(y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder


TOKEN_REGEX = r"\d+[./]\d+|\d+%|[a-zA-Z]+\-[a-zA-Z]+|[\w'’®]+|[.,!?;():+#&/%\-+]"

def tokenize(s, lemmatize=False, lowercase=True, normalize=False):
    if lowercase:
        s = s.lower()
    if normalize:
        s = normalizer(s)

    #if sys.version_info.major == 3:
    #    s = s.decode('utf-8')
    result = []
    for i in re.findall(TOKEN_REGEX, s, re.UNICODE):
        if lemmatize:
            i = lemmatizer(i)
        result.append(i)

    return result


class TokenizerWrapper(object):
    '''
    Creates a wrapper to Tokenizer of Keras.

        #Arguments
            lower: boolean. Whether to set the text to lowercase.
            split: str. Separator for word splitting.
            nb_words: None or int. Maximum number of words to work with (if set,
                tokenization will be restricted to the top nb_words most common
                words in the dataset.
            max_input_length: None or int. Maximum sequence length, longer
                sequences are truncated and shorter sequences are padded with
                zeros at the end.
    '''

    def __init__(self,
                 lower=True,
                 split=" ",
                 nb_words=None,
                 max_input_length=None,
                 char_level=False,
                 lemmatize=False,
                 normalize=False):
        self.lower = lower
        self.char_level = char_level
        self.encoder = Tokenizer(
            lower=lower,
            split=split,
            num_words=nb_words,
            char_level=char_level)
        self.nb_words = nb_words
        self.max_input_length = max_input_length
        self.lemmatize = lemmatize
        self.normalize = normalize


    def fit(self, labels=None):
        '''
        Fits encoder.
            #Arguments
                labels:  list.List containing all labels
        '''
        _labels = []

        for l in labels:
            aux = tokenize(l, self.lemmatize, self.lower, self.normalize)
            str_aux = ''
            for i in aux:
                str_aux = '{}{}'.format(str_aux, i)
            _labels.append(str_aux)
        self.encoder.fit_on_texts(_labels)

    def transform(self, labels=None):
        '''
        Encodes input.
            #Arguments
                labels:  list. List containing all labels
        '''
        _labels = []
        for l in labels:
            aux = tokenize(l, self.lemmatize, self.lower, self.normalize)
            str_aux = ''
            for i in aux:
                str_aux = '{}{}'.format(str_aux, i)
            _labels.append(str_aux)

        sequences = self.encoder.texts_to_sequences(_labels)
        return pad_sequences(sequences, maxlen=self.max_input_length)

