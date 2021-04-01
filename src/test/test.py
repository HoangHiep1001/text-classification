import codecs
import multiprocessing
from sklearn import utils
import regex as re
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors, Word2Vec

from src.preprocess.word2vec import read_data


def remove_stopword(text):
    filename = '../../data/stopwords.txt'
    list_stopwords = []
    with codecs.open(filename, "r", encoding="utf8") as file:
        a = file.readlines()
        for i in a:
            list_stopwords.append(i.strip())
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
        text2 = ' '.join(pre_text)
    return text2


if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065,
                             min_alpha=0.065)
    content, label = read_data()
    input_gensim = []

    for str in content:
        input_gensim.append(str.split())

    model_ug_cbow.build_vocab([x for x in tqdm(input_gensim)])
    for epoch in range(30):
        model_ug_cbow.train(utils.shuffle([x for x in tqdm(input_gensim)]), total_examples=len(input_gensim), epochs=1)
        model_ug_cbow.alpha -= 0.002
        model_ug_cbow.min_alpha = model_ug_cbow.alpha
    #init model
    model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065,
                           min_alpha=0.065)
    model_ug_sg.build_vocab([x for x in tqdm(input_gensim)])
    for epoch in range(30):
        model_ug_sg.train(utils.shuffle([x for x in tqdm(input_gensim)]), total_examples=len(input_gensim), epochs=1)
        model_ug_sg.alpha -= 0.002
        model_ug_sg.min_alpha = model_ug_sg.alpha
    model_ug_cbow.save('w2v_model_ug_sg.word2vec')
    model_ug_cbow.save('w2v_model_ug_cbow.word2vec')
