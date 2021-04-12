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
    list_folder = os.listdir(pathdata)
    for filename in list_folder:
        sents = open(pathdata + filename, 'r', encoding='utf-8').readlines()
        for sent in sents:
            content.append(sent)
            label.append(filename)
    return content, label


if __name__ == '__main__':
    content, label = read_data()
    input_gensim = []

    for str in content:
        input_gensim.append(str.split())
    word_model = Word2Vec(input_gensim, size=300, min_count=1, iter=10)
    word_model.wv.save_word2vec_format(pathModelBin, fvocab=None, binary=True)
    word_model.wv.save_word2vec_format(pathModelTxt, fvocab=None, binary=False)
    word_model.wv.save("../../model/word.model")
