import os
from os import listdir

import gensim
from gensim.models import KeyedVectors

from src.preprocess.preprocess import text_preprocess

sep = os.sep


def loadData(path_data):
    texts = []
    with open(path_data, 'r', encoding="utf-8") as f:
        all_of_it = f.read()
        texts = text_preprocess(all_of_it);
    return texts


if __name__ == '__main__':
    str = loadData('../../data/test.txt')
    print(str)