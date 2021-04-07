import os

from keras.layers import Dense, Embedding, Flatten,GlobalMaxPool1D
from keras.models import Sequential

sep = os.sep


def cnn_model(vocab_size, embedding_dim, maxlen,weights,output_shape):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=maxlen, weights=weights))
    model.add(Flatten())
    model.add(GlobalMaxPool1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def bi_lstm_model():
    print("Ab")

