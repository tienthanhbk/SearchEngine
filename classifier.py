import numpy as np
import pandas as pd
import convenion
import json
import glob

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

from sklearn.model_selection import train_test_split

from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from underthesea import word_tokenize


def train_model(X_train, y_train, validation_data):
    # Initialize the constructor
    model = Sequential()

    # Add an input layer
    model.add(Dense(200, activation='relu', input_shape=(400,)))

    # Add one hidden layer
    model.add(Dense(100, activation='relu'))

    # Add an output layer
    model.add(Dense(1, activation='sigmoid'))

    print('Training classify model..')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=validation_data, epochs=20, batch_size=1, verbose=1)

    model.save('model/simple_classify_model.h5')  # creates a HDF5 file 'my_model.h5'


def test_model():
    X_test = np.load('model/X_test.npy')
    y_test = np.load('model/y_test.npy')
    model = load_model('model/simple_classify_model.h5')

    y_pred = model.predict(X_test)
    print(y_pred[:10])
    score = model.evaluate(X_test, y_test, verbose=1)
    print(score)



def raw_data(path_regex):
    doc2vec_model = Doc2Vec.load('gensim/model/question.d2v')
    PATH_JUDGED_EXAMPLE = 'elastic/judged/train/17630743.json'

    X_raw = []
    y_raw = []

    path_judgeds = glob.glob(path_regex)

    for path_judged in path_judgeds:
        with open(path_judged, 'r') as f:
            judged_result = json.load(f)

            origin_question = judged_result['origin_question']
            org_q_vector = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(origin_question, format='text')))
            org_q_vector = np.array(org_q_vector)

            for hit in judged_result['hits']:
                judged_question = hit['question']

                judged_q_vector = doc2vec_model.infer_vector(
                    simple_preprocess(word_tokenize(judged_question, format='text')))
                judged_q_vector = np.array(judged_q_vector)

                arr_concat = np.concatenate((org_q_vector, judged_q_vector))

                label = 0
                if hit['relate_q_q'] == 1 or hit['relate_q_q'] == 2:
                    label = 1

                X_raw.append(arr_concat)
                y_raw.append(label)

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    return X_raw, y_raw


def save_raw_data():
    X_train, y_train = raw_data('./elastic/judged/train/*.json')
    X_dev, y_dev = raw_data('./elastic/judged/dev/*.json')
    X_test, y_test = raw_data('./elastic/judged/test/*.json')

    print('X_train len: ', len(X_train))
    print('y_train len: ', len(y_train))

    print('X_dev len: ', len(X_dev))
    print('y_dev len: ', len(y_dev))

    print('X_test len: ', len(X_test))
    print('y_test len: ', len(y_test))

    np.save('model/X_train.npy', X_train)
    np.save('model/X_dev.npy', X_dev)
    np.save('model/X_test.npy', X_test)

    np.save('model/y_train.npy', y_train)
    np.save('model/y_dev.npy', y_dev)
    np.save('model/y_test.npy', y_test)


def load_raw_data():
    X_train = np.load('model/X_train.npy')
    X_dev = np.load('model/X_dev.npy')
    X_test = np.load('model/X_test.npy')

    y_train = np.load('model/y_train.npy')
    y_dev = np.load('model/y_dev.npy')
    y_test = np.load('model/y_test.npy')

    print('X_train len: ', len(X_train))
    print('y_train len: ', len(y_train))

    print('X_dev len: ', len(X_dev))
    print('y_dev len: ', len(y_dev))

    print('X_test len: ', len(X_test))
    print('y_test len: ', len(y_test))

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def create_validation_data(X_dev, y_dev):
    validation_data = []
    for sample, label in zip(X_dev, y_dev):
        validation_data.append((sample, label))
    return validation_data

# Split the data up in train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
#
# train_model(X_train, y_train, X_test, y_test)


# X_train, X_dev, X_test, y_train, y_dev, y_test = load_raw_data()
#
# validation_data = create_validation_data(X_dev, y_dev)
#
# train_model(X_train, y_train, (X_dev, y_dev))
# test_model()
load_raw_data()
