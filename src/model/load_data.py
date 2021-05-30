import os
import pickle
from os import listdir

import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

sep = os.sep
max_length_sentence = 300
embeding_size = 300
vocab_size = 30000
classes = ['dien_anh', 'du_lich', 'suc_khoe', 'giao_duc', 'kinh_doanh', 'ngan_hang', 'the_thao', 'thoi_su_phap_luat']


def read_data(path_process):
    content = [], labels = []
    for folder in listdir(path_process):
        for file in listdir(path_process + sep + folder):
            with open(path_process + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                print("read file: " + path_process + sep + folder + sep + file)
                all_of_it = f.read()
                sentences = all_of_it.split('\n')
                for str in sentences:
                    content.append(str)
                for _ in sentences:
                    labels.append(classes.index(folder))
                del all_of_it, sentences
    return content, labels


def dump_data(content, labels):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[]^`{|}~ ', num_words=vocab_size)
    tokenizer.fit_on_texts(content)
    content_sequences = tokenizer.texts_to_sequences(content)
    pad_train_sequence = pad_sequences(content_sequences, maxlen=500, truncating='post', padding='post')

    y = pd.get_dummies(labels)
    file = open("../content/drive/MyDrive/project/data/data_dump/data.pkl", 'wb')
    pickle.dump([pad_train_sequence, y], file)
    file.close()


if __name__ == '__main__':
    content, labels = read_data('../../data/data_process')
    dump_data(content=content, labels=labels)
    print("=========done dump_data=========")
