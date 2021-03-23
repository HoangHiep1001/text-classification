import codecs

import pandas as pd
from gensim.models import KeyedVectors

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
    text = 'hiệp mai số và ai tính'
    print(remove_stopword(text))