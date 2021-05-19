import gensim.models.keyedvectors as word2vec
import matplotlib.pyplot as plt
from gensim.models.deprecated.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA

import os

# model = KeyedVectors.load_word2vec_format('../../model/baomoi.vn.model.bin', binary=True)
model = word2vec.KeyedVectors.load('../../w2v/word_model.save')
# model = word2vec.KeyedVectors.load('../model/fasttext_gensim.model')

pathfile = '../../model/words.txt'
with open(pathfile, 'r') as f:
    words = f.readlines()
    words = [word.strip().encode().decode('utf-8') for word in words]

words_np = []
words_label = []

for word in model.wv.vocab.keys():
    print(word)
    if word in words:
        words_np.append(model[word])
        words_label.append(word)

pca = PCA(n_components=2)
pca.fit(words_np)
reduced = pca.transform(words_np)


def visualize():
    fig, ax = plt.subplots()

    for index, vec in enumerate(reduced):
        x, y = vec[0], vec[1]

        ax.scatter(x, y)
        ax.annotate(words_label[index], xy=(x, y))

    plt.show()
    return


if __name__ == '__main__':
    visualize()