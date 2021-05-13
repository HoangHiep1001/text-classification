import os
import pickle

import gensim
import keras
import numpy as np
from gensim.models import Word2Vec
from keras.callbacks import *
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

from src.utils.plot_model import plot_model_history

sep = os.sep


def generate_model_w2v(data_folder):
    file = open(data_folder + sep + "data.pkl", 'rb')
    X, y, texts = pickle.load(file)
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
    # train Word2Vec model on our data
    # if not os.path.exists(data_folder + sep + "word_model.save"):
    input_gensin = []
    for str in texts:
        input_gensin.append(str.split())
    word_model = gensim.models.Word2Vec(input_gensin, size=300, min_count=1, iter=10)
    word_model.save(data_folder + sep + "word_model.save")
    word_model.wv.save_word2vec_format(data_folder + sep + 'test_w2v.txt', binary=False)
    # else:
    #     word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model.save")
    vocabulary_size = min(len(word_model.wv.vocab) + 1, 20000)
    embedding_matrix = np.zeros((vocabulary_size, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        if i > 20000:
            continue;
        try:
            embedding_matrix[i] = vec
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), 300)
    print(len(word_model.wv.vocab) + 1)
    if not os.path.exists(data_folder + sep + "predict_model.save"):
        # init layer
        model = Sequential()
        model.add(Embedding(vocabulary_size, 300, weights=[embedding_matrix], trainable=True))
        model.add(Bidirectional(LSTM(300, return_sequences=False)))
        model.add(Dropout(0.1))
        model.add(Dense(y.shape[1], activation="softmax"))
        model.summary()
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        # model.load_weights("../content/drive/MyDrive/project/data/data_dump/epochs:010-val_acc:0.948.hdf5")
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])
        # test checkpoint
        filepath = "../../data/data_dump/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            batch_size=64, epochs=10, verbose=1, callbacks=callbacks_list)
        model.save("predict_model_v2.h5")
        plot_model_history(history)
    else:
        model = load_model("predict_model_v2.h5")

    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    path = "../../data/data_dump/"
    generate_model_w2v(data_folder=path)
