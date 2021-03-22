from gensim.models import Word2Vec
import pandas as pd
import os

# path data
pathdata = '../../data/data_process/bat_dong_san.txt'
pathModelBin = '../../model/w2v.bin'
pathModelTxt = '../../model/w2v.txt'

def read_data(path):
    traindata = []
    sents = open(pathdata, 'r', encoding='utf-8').readlines()
    for sent in sents:
        traindata.append(sent.split(",")[0])
    return traindata


def readdata(path):
    list_file = os.listdir(path)
    data = pd.DataFrame()
    for filename in list_file:
        data = pd.concat([data, pd.read_csv(os.path.join(path, filename), sep=',')])

    return data.Review, data.Label

if __name__ == '__main__':
    path = "../../data/data_process.csv"
    data = pd.read_csv(path,sep=",", encoding='utf-8')['content']
    print(data)
    input_gensim = []
    for str in data:
        input_gensim.append(str.split())

    model = Word2Vec(input_gensim, size=200, window=3, min_count=1, sample=0.0001, workers=4, sg=0, negative=10,
                     cbow_mean=1,
                     iter=5)

    model.wv.save_word2vec_format(pathModelBin, fvocab=None, binary=True)
    model.wv.save_word2vec_format(pathModelTxt, fvocab=None, binary=False)
    model.wv.save("../../model/word.model")
