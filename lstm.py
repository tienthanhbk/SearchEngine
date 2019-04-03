import re
import nltk
import unicodedata
import string
import numpy as np
from underthesea import word_tokenize

from keras import backend as K
from keras.layers import Embedding
from keras.layers import LSTM, Input, dot, Lambda, GlobalMaxPool1D, Dense, Dropout, concatenate, CuDNNLSTM, BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import logging
from gensim.models import Word2Vec
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from tensorflow.contrib import learn
import random


import logging
from gensim.models import Word2Vec
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from tensorflow.contrib import learn
import random


# from google.colab import drive
# drive.mount('/content/driver/')

import nltk
nltk.download('punkt')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def url_elimination(text):
    urls = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', text)
    output = ''
    for url in urls:
        x = text.find(url)
        if x > 0:
            output += text[:x]
            output += "url "
            text = text[x+len(url) +1:]
    output += text
    return output


def tokenize(text) :
    # text = url_elimination(text)
    return [w.lower() for w in word_tokenize(text, format='text')]


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_punctuation2(words):
    new_words = []
    for word in words:
        temp = word.strip(string.punctuation)
        if temp is not '':
            new_words.append(temp)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    return [re.sub(r'\d+', '', word) for word in words]


def clean(text):
    # text = clean_str(text)
    words = tokenize(text)
    # words = remove_non_ascii(words)
    # words = remove_punctuation2(words)
    # words = replace_numbers(words)
    return ' '.join(words)


def get_modified_data(FILE_PATH):
    f = open(FILE_PATH, 'r')
    data_processed = []
    for line in f.readlines():
        line = line.strip()
        temp = line.split('\t')
        for i in range(2):
            temp[i] = clean(temp[i])
        data_processed.append(temp)
    f.close()
    print(data_processed[:10])
    return data_processed


def build_corpus(FILE_PATH):
    data_processed = get_modified_data(FILE_PATH)
    questions = []
    answers = []
    labels = []
    for i in range(len(data_processed)):
        questions.extend([data_processed[i][0]])
        answers.append([data_processed[i][1]])
        labels.append(int(data_processed[i][2]))
    return questions, answers, labels


def sentence_to_vec(sentence, vocab):
    splited_sentence = sentence.split(' ')
    result = np.zeros([len(splited_sentence), ], dtype=int)
    for i in range(len(splited_sentence)):
        if splited_sentence[i] in vocab:
            result[i] = get_index(splited_sentence[i], vocab)
        else:
            result[i] = random.randint(0, len(vocab))
    return result


def turn_to_vector(list_to_transform, vocab):
    # vocab_size = 44604
    # pad = 150
    encoded_list = [sentence_to_vec(str(d), vocab) for d in list_to_transform]
    padded_list = pad_sequences(encoded_list, maxlen=150, padding='post', truncating='post')
    return padded_list


def get_index(word, vocab):
    return vocab[word]
#
# def create_model(FILE_PATH):
#     prep = preprocess.PreprocessData()
#     data_processed = prep.get_modified_data(FILE_PATH)
#     train_data = []
#     for data_point in data_processed:
#         train_data.append(data_point[0])
#         train_data.append(data_point[1])
#     # print(train_data)
#     logging.basicConfig(
#         format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#     model = Word2Vec(train_data, size=100, window=10,
#                      min_count=1, workers=4, sg=0)
#     filename = 'cbow_model.pkl'
#     pickle.dump(model, open(filename, 'wb'))


def create_vocab_with_index(model):
    with open('/content/driver/My Drive/DATN/vocab_all.txt', 'w') as f:
        vocab = model.wv.vocab
        for key, _ in vocab.items():
            index = vocab[key].index
            f.write(str(index) + '\t' + key)
            f.write('\n')
    f.close()


def create_vocab_dict():
    vocab = {}
    with open('data/lstm/vocab_all.txt', 'r') as f:
        for line in f.readlines():
            temp = line.split('\t')
            vocab[temp[1].strip()] = temp[0].strip()
    f.close()
    return vocab


def get_glove_vectors():
    # N = 200
    g = dict()
    file_path = 'data/lstm/vectors.txt'

    with open(file_path, 'r') as f:
        for line in f.readlines():
            temp = line.split()
            word = temp[0]
            g[word] = np.array(temp[1:]).astype(float)
    return g


def embmatrix(g, vocab):
    embedding_weights = np.zeros((len(vocab) + 1, 200), dtype=float)
    for word in vocab.keys():
        if word in g:
            embedding_weights[int(vocab[word]), :] = np.array(g[word])
        else:
            embedding_weights[int(vocab[word]), :] = np.random.uniform(-1, 1, 200)
    return embedding_weights


def map_score(s1s_dev, s2s_dev, y_pred, labels_dev):
    QA_pairs = {}
    for i in range(len(s1s_dev)):
        pred = y_pred[i]

        s1 = str(s1s_dev[i])
        s2 = str(s2s_dev[i])
        if s1 in QA_pairs:
            QA_pairs[s1].append((s2, labels_dev[i], pred))
        else:
            QA_pairs[s1] = [(s2, labels_dev[i], pred)]

    MAP, MRR = 0, 0
    num_q = len(QA_pairs.keys())
    for s1 in QA_pairs.keys():
        p, AP = 0, 0
        MRR_check = False

        QA_pairs[s1] = sorted(
            QA_pairs[s1], key=lambda x: x[-1], reverse=True)

        for idx, (s2, label, _) in enumerate(QA_pairs[s1]):
            if int(label) == 1:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True

                p += 1
                AP += p / (idx + 1)
        if p == 0:
            AP = 0
        else:
            AP /= p
        MAP += AP
    MAP /= num_q
    MRR /= num_q
    return MAP, MRR


class AnSelCB(Callback):
    def __init__(self, val_q, val_s, y, inputs):
        self.val_q = val_q
        self.val_s = val_s
        self.val_y = y
        self.val_inputs = inputs

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.val_inputs)
        map__, mrr__ = map_score(self.val_q, self.val_s, pred, self.val_y)
        print('val MRR %f; val MAP %f' % (mrr__, map__))
        logs['mrr'] = mrr__
        logs['map'] = map__


def get_bilstm_model(vocab_size, vocab):
    enc_timesteps = 150
    dec_timesteps = 150
    # hidden_dim = 128

    question = Input(shape=(enc_timesteps,),
                     dtype='int32', name='question_base')
    answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')

    g = get_glove_vectors()
    weights = embmatrix(g, vocab)
    qa_embedding = Embedding(
        input_dim=vocab_size + 1, input_length=150, output_dim=weights.shape[1], mask_zero=False, weights=[weights])
    bi_lstm = Bidirectional(
        CuDNNLSTM(units=750, return_sequences=False))

    question_embedding = qa_embedding(question)
    question_embedding = Dropout(0.75)(question_embedding)
    question_enc_1 = bi_lstm(question_embedding)
    question_enc_1 = Dropout(0.75)(question_enc_1)
    question_enc_1 = BatchNormalization()(question_enc_1)

    answer_embedding = qa_embedding(answer)
    answer_embedding = Dropout(0.75)(answer_embedding)
    answer_enc_1 = bi_lstm(answer_embedding)
    answer_enc_1 = Dropout(0.75)(answer_enc_1)
    answer_enc_1 = BatchNormalization()(answer_enc_1)

    qa_merged = concatenate([question_enc_1, answer_enc_1])
    qa_merged = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0007))(qa_merged)
    qa_merged = Dropout(0.75)(qa_merged)
    qa_merged = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001),
                      activity_regularizer=regularizers.l1(0.001))(qa_merged)
    #         qa_merged = Dense(1, activation='sigmoid')(qa_merged)
    lstm_model = Model(name="bi_lstm", inputs=[
        question, answer], outputs=qa_merged)
    output = lstm_model([question, answer])
    training_model = Model(
        inputs=[question, answer], outputs=output, name='training_model')
    opt = Adam(lr=0.0001)
    training_model.compile(loss='binary_crossentropy', optimizer=opt)
    return training_model


def train():
    vocab = create_vocab_dict()
    vocab_len = len(vocab)

    training_model = get_bilstm_model(vocab_len, vocab)

    questions, answers, labels = build_corpus('/content/driver/My Drive/DATN/train.txt')

    q_dev, a_dev, l_dev = build_corpus('/content/driver/My Drive/DATN/dev.txt')

    # questions, answers = turn_to_vector(questions, answers, tok)
    # q_dev_eb, a_dev_eb = turn_to_vector(q_dev, a_dev, tok)
    questions = turn_to_vector(questions, vocab)
    answers = turn_to_vector(answers, vocab)
    q_dev_eb = turn_to_vector(q_dev, vocab)
    a_dev_eb = turn_to_vector(a_dev, vocab)
    Y = np.array(labels)
    callback_list = [AnSelCB(q_dev, a_dev, l_dev, [q_dev_eb, a_dev_eb]),
                     ModelCheckpoint('model_CuDNNimprovement-{epoch:02d}-{map:.2f}.h5', monitor='map', verbose=1, save_best_only=True, mode='max'),
                     EarlyStopping(monitor='map', mode='max', patience=22)]
    training_model.fit(
        [questions, answers],
        Y,
        epochs=100,
        batch_size=80,
        validation_data=([q_dev_eb, a_dev_eb], l_dev),
        verbose=1,
        callbacks=callback_list
    )
    training_model.summary()


def test_model():
    vocab = create_vocab_dict()
    vocab_len = len(vocab)

    training_model = get_bilstm_model(vocab_len, vocab)
    training_model.load_weights('model/bi-lstm/model_CuDNNimprovement-01-0.28.h5')

    questions, answers, labels = build_corpus('data/lstm/test_lstm.txt')
    print(len(questions))

    questions_eb = turn_to_vector(questions, vocab)
    answers_eb = turn_to_vector(answers, vocab)

    # Y = np.array(labels)

    sims = training_model.predict([questions_eb, answers_eb])

    MAP, MRR = map_score(questions, answers, sims, labels)
    print("MAP: ", MAP)
    print("MRR: ", MRR)


test_model()