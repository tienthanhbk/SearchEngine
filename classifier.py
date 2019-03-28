import numpy as np
import pandas as pd
import convenion
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

from sklearn.model_selection import train_test_split

from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from underthesea import word_tokenize


def train_model(X_train, y_train, X_test, y_test):
    # Save data to file
    np.save('model/X_train', X_train)
    np.save('model/X_test', X_test)
    np.save('model/y_train', y_train)
    np.save('model/y_test', y_test)

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

    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

    y_pred = model.predict(X_test)
    print(y_pred[:10])
    score = model.evaluate(X_test, y_test, verbose=1)

    print(score)
    model.save('model/simple_classify_model.h5')  # creates a HDF5 file 'my_model.h5'

def raw_data():
    PATH_JUDGED = 'elastic/judged/tmp/17630743.json'

    doc2vec_model = Doc2Vec.load('gensim/model/question.d2v')

    with open(PATH_JUDGED, 'r') as f:
        judged_result = json.load(f)

        origin_question = judged_result['origin_question']
        arr_origin_q = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(origin_question, format='text')))
        arr_origin_q = np.array(arr_origin_q)
        # print('arr_origin_q: ', arr_origin_q)
        X = []
        y = []
        for hit in judged_result['hits']:
            judged_question = hit['question']

            arr_judged_q = doc2vec_model.infer_vector(simple_preprocess(word_tokenize(judged_question, format='text')))
            arr_judged_q = np.array(arr_judged_q)

            arr_concat = np.concatenate((arr_origin_q, arr_judged_q))

            label = 0
            if hit['relate_q_q'] == 1 or hit['relate_q_q'] == 2:
                label = 1

            X.append(arr_concat)
            y.append(label)

        X = np.array(X)
        return X, y


X, y = raw_data()

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

train_model(X_train, y_train, X_test, y_test)
