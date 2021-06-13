import pickle
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, Dropout,Flatten
from keras.callbacks import *
import gensim
import keras
import numpy as np
from gensim.models import Word2Vec
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.utils.plot_model import plot_model_history

sep = os.sep



def train_model():
    file = open("../../data/data_train.pkl", 'rb')
    X, y = pickle.load(file)
    file.close()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
    # load w2v model
    word_model = gensim.models.Word2Vec.load("../../data/crawl/word_model.save")
    vocabulary_size = min(len(word_model.wv.vocab) + 1, 40000)
    embedding_matrix = np.zeros((vocabulary_size, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        if i >= 30000:
            continue;
        try:
            embedding_matrix[i] = vec
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 300)
    #init model
    model = Sequential()
    model.add(Embedding(vocabulary_size, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(LSTM(300, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(y.shape[1], activation="softmax"))
    opt = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])

    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)
    early = EarlyStopping(monitor='val_loss')
    filepath = "../content/drive/MyDrive/project/data/data_dump/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=32, epochs=10, verbose=1, callbacks=callbacks_list)
    model.save("../content/drive/MyDrive/project/data/model/model_predict/predict_model_bi_lstm.h5")
    plot_model_history(history)


if __name__ == '__main__':
    path = "../content/drive/MyDrive/project/data/data_dump"
    train_model()
