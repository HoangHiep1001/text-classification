import gensim.models.keyedvectors as word2vec
from matplotlib import pyplot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path_model = '../../model/word.model'
model = word2vec.KeyedVectors.load(path_model)
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])


words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()