import pandas as pd
from gensim.models import KeyedVectors
if __name__ == '__main__':
    model = KeyedVectors.load('../../model/word.model')

    for word in model.most_similar(u"tôi"):
        print(word[0])