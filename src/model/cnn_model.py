import os
import pickle

import gensim
import numpy as np
from keras.callbacks import *
from keras.layers import Input, Dense, Embedding, Dropout, Bidirectional, LSTM
from keras.layers.core import Flatten
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from src.utils.plot_model import plot_model_history

sep = os.sep


def train_model(data_folder):
    file = open("/data.pkl", 'rb')
    X, y = pickle.load(file)
    file.close()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
    word_model = gensim.models.Word2Vec.load("/home/hiephm/PycharmProjects/text-classification/w2v/word_model.save")
    vocabulary_size = min(len(word_model.wv.vocab) + 1, 30000)
    embedding_matrix = np.zeros((vocabulary_size, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        if i >= 30000:
            continue;
        try:
            embedding_matrix[i] = vec
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 300)

    model = Sequential()
    model.add(Embedding(30000, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu', strides=1))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    early = EarlyStopping(monitor='val_loss')
    filepath = "../content/drive/MyDrive/project/data/data_dump/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early]
    history = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_val, y_val),
                        shuffle=True, callbacks=callbacks_list)
    model.save(data_folder + sep + "predict_model.h5")
    plot_model_history(history)

def train_model_bi_lstm():
    file = open("../../data/data.pkl", 'rb')
    X, y = pickle.load(file)
    print(X.shape)
    print(X)
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
    word_model = gensim.models.Word2Vec.load("../../data/model/word_model.save")
    vocabulary_size = min(len(word_model.wv.vocab) + 1, 30000)
    embedding_matrix = np.zeros((vocabulary_size, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        if i >= 30000:
            continue;
        try:
            embedding_matrix[i] = vec
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 300)
    print(len(word_model.wv.vocab) + 1)
    model = Sequential()
    model.add(Embedding(30000, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(300, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])
    # test checkpoint
    early = EarlyStopping(monitor='val_loss')
    filepath = "../content/drive/MyDrive/project/data/data_dump/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,early]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=64, epochs=20, verbose=1, callbacks=callbacks_list)
    model.save("../content/drive/MyDrive/project/data/model/model_predict/predict_model_bi_lstm.h5")
    plot_model_history(history)
    model.evaluate(X_test, y_test)
def train_model_lstm():
    file = open("../../data/data.pkl", 'rb')
    X, y = pickle.load(file)
    print(X.shape)
    print(X)
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
    word_model = gensim.models.Word2Vec.load("../../data/model/word_model.save")
    vocabulary_size = min(len(word_model.wv.vocab) + 1, 30000)
    embedding_matrix = np.zeros((vocabulary_size, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        if i >= 30000:
            continue;
        try:
            embedding_matrix[i] = vec
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 300)
    print(len(word_model.wv.vocab) + 1)
    model = Sequential()
    model.add(Embedding(30000, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(LSTM(300, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])
    # test checkpoint
    early = EarlyStopping(monitor='val_loss')
    filepath = "../content/drive/MyDrive/project/data/data_dump/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,early]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=64, epochs=20, verbose=1, callbacks=callbacks_list)
    model.save("../content/drive/MyDrive/project/data/model/model_predict/predict_model_bi_lstm.h5")
    plot_model_history(history)
    model.evaluate(X_test, y_test)
if __name__ == '__main__':
    train_model_cnn()
