import os

import gensim
from gensim.models import Word2Vec

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
    word_model = gensim.models.Word2Vec(input_gensim, size=300, min_count=1, iter=10, sample=0.00001,window=5,cbow_mean=1)
    word_model.save("../../data/word_model.save")
    word_model.wv.save_word2vec_format('../../data/w2v_model.txt', binary=False)
