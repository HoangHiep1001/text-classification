import pickle
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, Dropout,Flatten,Conv1D,GlobalMaxPooling1D
from keras.callbacks import *
import gensim
import keras
import numpy as np
from gensim.models import Word2Vec
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sep = os.sep


def plot_model_history(model_history, acc='acc', val_acc='val_acc'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    plt.savefig('roc.png')


def train_model_cnn():
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
    model.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])
    check_point_path = "../../data/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(check_point_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=64, epochs=20, verbose=1, callbacks=callbacks_list)
    model.save("../../data/model/predict_model_lstm.h5")
    plot_model_history(history)

    model.evaluate(X_test, y_test)
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
