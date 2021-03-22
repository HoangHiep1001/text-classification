from gensim.models import Word2Vec
import pandas as pd

# path data
pathdata = './data/data-test.txt'


def read_data(path):
    traindata = []
    sents = open(pathdata, 'r', encoding='utf-8').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata


pathModelBin = 'model/w2v.bin'
pathModelTxt = 'model/w2v.txt'

if __name__ == '__main__':
    data = pd.read_csv('../../data/data_process.csv', sep=",", encoding='utf-8');
    train_data = data['content']

    try:
        model = Word2Vec(train_data, size=100, window=3, min_count=1, sample=0.0001, workers=4, sg=0, negative=10,
                         cbow_mean=1,
                         iter=5)
        model.wv.save_word2vec_format(pathModelBin, fvocab=None, binary=True)
        model.wv.save_word2vec_format(pathModelTxt, fvocab=None, binary=False)
    except:

        print("error")

    print("\nTrain done saved to ./model folder.")
