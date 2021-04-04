import pickle
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding
import gensim
import numpy as np
from gensim.models import Word2Vec
import os
from sklearn.model_selection import train_test_split
sep = os.sep


def generate_model_w2v(data_folder):
    file = open(data_folder + sep + "data.pkl", 'rb')
    X,y,texts = pickle.load(file)
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    # train Word2Vec model on our data
    if not os.path.exists(data_folder + sep + "word_model.save"):
        word_model = gensim.models.Word2Vec(texts, size=300, min_count=1, iter=10)
        word_model.save(data_folder + sep + "word_model.save")
    else:
        word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model.save")
    # split in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

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
        model = Sequential()
        model.add(Embedding(len(word_model.wv.vocab) + 1, 300, input_length=X.shape[1], weights=[embedding_matrix],
                            trainable=False))
        model.add(LSTM(300, return_sequences=False))
        model.add(Dense(y.shape[1], activation="softmax"))
        model.summary()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])

        batch = 64
        epochs = 1
        model.fit(X_train, y_train, batch, epochs)
        model.save(data_folder + sep + "predict_model.save")
    else:
        model = load_model("predict_model.save")

    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    path = "../../data/data_dump/"
    generate_model_w2v(path)
