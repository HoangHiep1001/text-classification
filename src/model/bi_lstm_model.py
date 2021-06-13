import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from keras.callbacks import *
import gensim
import keras
import numpy as np
from gensim.models import Word2Vec
import os
from sklearn.model_selection import train_test_split
from src.utils.plot_model import plot_model_history

sep = os.sep

VOCAB_SIZE = 40000


def train_model():
    #load data from file dump
    file = open('../../data/data_train.pkl', 'rb')
    X, y = pickle.load(file)
    file.close()
    #split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)
    # load pre-trained w2v model from file
    word_model = gensim.models.Word2Vec.load("../../data/crawl/word_model.save")
    vocabulary_size = min(len(word_model.wv.vocab) + 1, VOCAB_SIZE)
    embedding_matrix = np.zeros((vocabulary_size, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        if i >= vocabulary_size:
            continue;
        try:
            embedding_matrix[i] = vec
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 300)
    print(vocabulary_size)
    #init model
    model = Sequential()
    model.add(Embedding(vocabulary_size, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=True))
    model.add(Bidirectional(LSTM(300, return_sequences=True)))
    model.add(Bidirectional(LSTM(300)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='bi-lstm_model.png', show_shapes=True, show_layer_names=True)
    opt = keras.optimizers.Adam(learning_rate=10e-6)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])
    early = EarlyStopping(monitor='val_loss')
    path_checkpoint = "../../data/check_point/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(path_checkpoint, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=20, verbose=1, callbacks=callbacks_list)
    model.save("../../model/predict_model_bi_lstm.h5")
    plot_model_history(history)
    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    train_model()
