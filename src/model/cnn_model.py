import pickle
from tensorflow.keras import layers
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding,Bidirectional,Dropout,Dropout
import gensim
import keras
import numpy as np
from gensim.models import Word2Vec
import os
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import *
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers

filter_sizes = [3,4,5]
num_filters = 100
sep = os.sep


def generate_model_w2v(data_folder):
    file = open(data_folder + sep + "data.pkl", 'rb')
    X,y,texts = pickle.load(file)
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
    # train Word2Vec model on our data
    if not os.path.exists(data_folder + sep + "word_model.save"):
        word_model = gensim.models.Word2Vec(texts, size=300, min_count=1, iter=10)
        word_model.save(data_folder + sep + "word_model.save")
    else:
        word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model.save")
    # check the most similar word to 'cơm'
    print(word_model.wv.most_similar('công_điện'))

    embedding_matrix = np.zeros((len(word_model.wv.vocab) + 1, 300))
    for i, vec in enumerate(word_model.wv.vectors):
        embedding_matrix[i] = vec

    if not os.path.exists(data_folder + sep + "predict_model.save"):
        # init layer
        EMBEDDING_DIM = 300
        sequence_length = X_train.shape[1]
        embedding_layer = Embedding(len(word_model.wv.vocab) + 1,300,
                            weights=[embedding_matrix],
                            trainable=True)
        inputs = Input(shape=(sequence_length,))
        embedding = embedding_layer(inputs)
        reshape = Reshape((sequence_length,300,1))(embedding)

        conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
        conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
        conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

        maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
        maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
        maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

        merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
        flatten = Flatten()(merged_tensor)
        reshape = Reshape((3*num_filters,))(flatten)
        dropout = Dropout(0.1)(flatten)
        output = Dense(units=y_train.shape[1], activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)
        model = Model(inputs, output)
        adam = Adam(lr=1e-3)
        model.load_weights("../content/drive/MyDrive/project/data/data_dump/epochs:001-val_acc:0.826.hdf5")
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc'])
        print(model.summary())
        early = EarlyStopping(monitor='val_loss')
        #test checkpoint
        filepath = "../content/drive/MyDrive/project/data/data_dump/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint,early]
        history = model.fit(X_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(X_val, y_val),shuffle=True,
                callbacks=callbacks_list)
        model.save(data_folder + sep + "predict_model.h5")
        # plot_model_history(history)
    else:
        model = load_model(data_folder + sep + "predict_model.h5")

    model.evaluate(X_test,y_test)


if __name__ == '__main__':
    path = "../content/drive/MyDrive/project/data/data_dump/"
    generate_model_w2v(path)