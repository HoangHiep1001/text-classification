import random
import numpy as np
from gensim.models import Word2Vec
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot

# path data
pathdata = '../../data/data_process/'
pathModelBin = '../../model/w2v.bin'
pathModelTxt = '../../model/w2v.txt'


def read_data():
    content = []
    label = []
    list_file = os.listdir(pathdata)
    for filename in list_file:
        sents = open(pathdata + filename, 'r', encoding='utf-8').readlines()
        for sent in sents:
            content.append(sent)
            label.append(filename)
    l = len(content)
    shutfle = list(range(l))
    random.shuffle(shutfle)
    train_data = np.array(content)
    train_label = np.array(label)
    train_data = train_data[shutfle]
    train_label = train_label[shutfle]

    return train_data, train_label


if __name__ == '__main__':
    content, label = read_data()
    input_gensim = []

    for str in content:
        input_gensim.append(str.split())

    model = Word2Vec(input_gensim, size=250, window=5, min_count=1, workers=4, sg=1,iter=5)

    model.wv.save_word2vec_format(pathModelBin, fvocab=None, binary=True)
    model.wv.save_word2vec_format(pathModelTxt, fvocab=None, binary=False)
    model.wv.save("../../model/word.model")
