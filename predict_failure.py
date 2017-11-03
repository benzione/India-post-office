import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
from itertools import chain
import math
import random
import gc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras import backend as K

np.random.seed(7)


def check():
    df = pd.read_csv('./Data/TRAINING_ARRAY_Upsampled_failures.csv', skiprows=2, header=None)
    countzero = 0
    countone = 0
    for i, row in enumerate(df.values):
        if row[33] == 0 and row[34] == 0:
            countzero += 1
        elif row[33] == 1 and row[34] == 1:
            countone += 1
    print countone
    print countzero


def read_files(name1, name2):
    first_file = pd.read_csv(name1, skiprows=1, header=None)
    second_file = pd.read_csv(name2, skiprows=2, header=None)
    list_address = []
    for i, row1 in enumerate(first_file.values):
        for j, row2 in enumerate(second_file.values):
            if row1[0] == row2[2]:
                strtmp = ''
                tmp = [row1[0]]
                for item in row1[1:]:
                    if type(item) is int or type(item) is float:
                        tmp.append(strtmp)
                        tmp.append(row2[33])
                        break
                    else:
                        strtmp += item + ' '
                list_address.append(tmp)
                break

    thefile = open('./Data/address.txt', 'w')
    for line in list_address:
        for item in line:
            thefile.write("%s\t" % item)
        thefile.write("\n")


def run_model(X_train, y_train, X_validate, y_validate, parameters, max_review_length, top_words):
    np.random.seed(3)

    model = Sequential()
    model.add(Embedding(top_words, int(parameters[0]), input_length=max_review_length))
    model.add(Dropout(float(parameters[1])))
    model.add(LSTM(int(parameters[2])))
    model.add(Dropout(float(parameters[3])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=int(parameters[4]),
              batch_size=int(parameters[5]), verbose=0,
              shuffle=True)

    # Final evaluation of the model
    scores = model.evaluate(X_validate, y_validate, verbose=0)

    model = X_train = y_train = X_validate = y_validate = parameters = None
    gc.collect()
    return scores[1] * 100


def run_optimize(cv, text, items, lst_labels_true, parameters, max_review_length, top_words):
    n = np.max(items)
    i = 0
    results = np.zeros([n, 1])
    for train_valid in xrange(n + 1):
        if cv != train_valid:
            X_train = [text[j] for j, val in enumerate(items) if val != train_valid and val != cv]
            y_train = [lst_labels_true[j] for j, val in enumerate(items) if val != train_valid and val != cv]

            X_validate = [text[j] for j, val in enumerate(items) if val == train_valid]
            y_validate = [lst_labels_true[j] for j, val in enumerate(items) if val == train_valid]

            # print("Accuracy: %.2f%%" % (scores[1]*100))
            results[i, 0] = run_model(X_train, y_train, X_validate, y_validate, parameters[0], max_review_length,
                                      top_words)
            print 'cv %d, iteration %d, and result = %.2f' % (cv, i, results[i, 0])
            i += 1

            X_train = y_train = X_validate = y_validate = None
    cv = text = items = lst_labels_true = parameters = n = i = None
    K.clear_session()
    gc.collect()
    return np.mean(results, axis=0)


def prepare_sequences(k ,k2):
    dataframe = pd.read_csv('./Data/address.txt', sep="\t", header=None)
    data = dataframe.values
    lst_labels_true = data[:, 2]
    sequences = data[:, 1]
    list_words = []
    max_review_length = 0
    for i, item in enumerate(sequences):
        if isinstance(item, basestring):
            tmp = CountVectorizer().build_tokenizer()(item)
            tmplen = len(tmp)
            if tmplen > max_review_length:
                max_review_length = tmplen
            list_words.append(tmp)
        else:
            np.delete(lst_labels_true, i, 0)

    totals = Counter(i for i in list(chain.from_iterable(list_words)))

    word_index = {}
    i = 1
    for item in totals.keys():
        if totals[item] > k2:
            word_index[item] = i
            i += 1

    text = []
    for row in list_words:
        tmp = []
        for word in row:
            if k2 < totals[word] < max(totals.values()):
                tmp.append(word_index[word])
            else:
                tmp.append(0)
        while len(tmp) < max_review_length:
            tmp.append(0)
        text.append(tmp)

    totals = Counter(i for i in list(chain.from_iterable(text)))
    top_words = len(totals.keys())

    # len_data_set = len(text)
    # items = [i for i in xrange(k)] * int(math.ceil(len_data_set / float(k)))
    # items = items[:len_data_set]
    # random.shuffle(items)
    #
    # str_file_new = './Data/cross_validation.txt'
    # the_file = open(str_file_new, 'w')
    # for element in items:
    #     the_file.write("%s\n" % element)
    # the_file.close()

    items = []
    with open('./Data/cross_validation.txt', 'r') as f:
        for line in f:
            tmp = line.strip('\n')
            items.append(int(tmp))

    class1 = 0
    class2 = 0
    for item in lst_labels_true:
        if item == 0:
            class1 += 1
        else:
            class2 += 1

    embedding_arr = xrange(10, 30, 5)
    dropout_arr1 = [i / float(10) for i in xrange(0, 10, 2)]
    lstm_hidden = xrange(500, 1301, 50)
    dropout_arr2 = [i / float(10) for i in xrange(0, 10, 2)]
    epoch = 35
    batch = 64

    n = k
    results_cv = np.zeros([n, 1])
    parameters = np.array([[800, 0.1, 750, 0.1, epoch, batch]])
    # parameters = np.load('./Data/Results_RNN/lstm2_parameters_cv0.npy')
    for cv in xrange(n):
        for i in xrange(2):
            results = np.zeros([len(embedding_arr), 1])
            for ii, embedding in enumerate(embedding_arr):
                print 'lstm optimize #%d embedding %d' % (i, ii)
                parameters[0, 0] = embedding
                results[ii, :] = run_optimize(cv, text, items, lst_labels_true, parameters, max_review_length,
                                              top_words)

            results = results[~np.all(results == 0, axis=1)]
            tmp = np.argmax(results, axis=0)
            parameters[0, 0] = embedding_arr[tmp[0]]
            print parameters

            results = np.zeros([len(dropout_arr1), 1])
            for ii, dropout in enumerate(dropout_arr1):
                print 'lstm optimize #%d dropout1 %d' % (i, ii)
                parameters[0, 1] = dropout
                results[ii, :] = run_optimize(cv, text, items, lst_labels_true, parameters, max_review_length,
                                              top_words)

            results = results[~np.all(results == 0, axis=1)]
            tmp = np.argmax(results, axis=0)
            parameters[0, 1] = dropout_arr1[tmp[0]]
            print parameters

            results = np.zeros([len(lstm_hidden), 1])
            for ii, hidden in enumerate(lstm_hidden):
                print 'lstm optimize #%d hidden %d' % (i, ii)
                parameters[0, 2] = hidden
                results[ii, :] = run_optimize(cv, text, items, lst_labels_true, parameters, max_review_length,
                                              top_words)

            results = results[~np.all(results == 0, axis=1)]
            tmp = np.argmax(results, axis=0)
            parameters[0, 2] = lstm_hidden[tmp[0]]
            print parameters

            results = np.zeros([len(dropout_arr2), 1])
            for ii, dropout in enumerate(dropout_arr2):
                print 'lstm optimize #%d dropout2 %d' % (i, ii)
                parameters[0, 3] = dropout
                results[ii, :] = run_optimize(cv, text, items, lst_labels_true, parameters, max_review_length,
                                              top_words)
            results = results[~np.all(results == 0, axis=1)]
            tmp = np.argmax(results, axis=0)
            parameters[0, 3] = dropout_arr2[tmp[0]]
            print parameters

        # print 'lstm test'
        # results_cv[cv, :] = run_test(cv, text, items, lst_labels_true, parameters)
        # gc.collect()
        np.save('./Data/Results_RNN/lstm2_parameters_cv%d.npy' % cv, parameters)
    #     print parameters
    #     print 'lstm CV %d resutls = %.2f' % (cv, results_cv[cv, :])
    # np.save('./Data/Results_RNN/lstm2_results.npy', results_cv)


def main():
    # name1 = './Data/INPUT_DATA_Upsampled_failures.csv'
    # name2 = './Data/TRAINING_ARRAY_Upsampled_failures.csv'
    # read_files(name1, name2)
    prepare_sequences(5, 15)
    # check()


if __name__ == '__main__':
    main()