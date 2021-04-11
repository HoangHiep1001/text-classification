import pickle
from os import listdir
import os
import gensim
import keras
import numpy as np
from gensim.models import Word2Vec
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from src.preprocess.preprocess import text_preprocess, text_preprocessV2
import pandas as pd
sep = os.sep
max_length_sentence = 300
embeding_size = 300
vocab_size = 5000


def read_data(path_raw):
    content = []
    labels = []
    texts = []
    for folder in listdir(path_raw):
        for file in listdir(path_raw + sep + folder):
            with open(path_raw + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                print("read file: " + path_raw + sep + folder + sep + file)
                all_of_it = f.read()
                sentence = all_of_it.split('.')
                sentence = text_preprocessV2(sentence)
                texts = texts + sentence
                sentences = all_of_it.split('\n')
                for str in sentences:
                    str = text_preprocess(str)
                    content.append(str)
                for _ in sentences:
                    labels.append(folder)
                del all_of_it, sentences
    return content, labels, texts


def process_with_keras(content):
    tokenize = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[]^`{|}~ ',num_words=vocab_size,oov_token='<OOV')
    tokenize.fit_on_texts(content)
    content_sequences = tokenize.texts_to_sequences(content)
    pad_train_sequence = pad_sequences(content_sequences,maxlen=max_length_sentence,truncating='post',padding='post')


def dump_data(data_folder):
    content, labels, texts = read_data(data_folder)
    tokenize = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[]^`{|}~ ',num_words=vocab_size,oov_token='<OOV>')
    tokenize.fit_on_texts(content)
    content_sequences = tokenize.texts_to_sequences(content)
    pad_train_sequence = pad_sequences(content_sequences,maxlen=300,truncating='post',padding='post')

    y = pd.get_dummies(labels)
    print(pad_train_sequence.shape)
    file = open("../../data/data_dump/data.pkl", 'wb')
    pickle.dump([pad_train_sequence,y, texts],file)
    file.close()


if __name__ == '__main__':
    path_data = '../../data/data_process'
    content, labels, texts = read_data(path_data)
    tokenize = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[]^`{|}~ ', num_words=vocab_size, oov_token='<OOV>')
    tokenize.fit_on_texts(content)
    word_index = tokenize.word_index
    content_sequences = tokenize.texts_to_sequences(content)
    pad_train_sequence = pad_sequences(content_sequences, maxlen=300, truncating='post', padding='post')
    print(word_index)
    y = pd.get_dummies(labels)
    print(pad_train_sequence.shape)
    file = open("../../data/data_dump/data.pkl", 'wb')
    pickle.dump([pad_train_sequence, y, texts], file)
    file.close()
    file = open("../../data/data_dump/data.pkl", 'rb')
    X, y, texts = pickle.load(file)
    print(X[0])
    file.close()
    word_model = Word2Vec(texts, size=300, window=2, min_count=1, sample=0.0001, workers=4, sg=0, negative=10,
                          cbow_mean=1, iter=5)
    word_model.save("word_model_v2.save")
    word_model.wv.save_word2vec_format('test_w2v_v2.txt', binary=False)
