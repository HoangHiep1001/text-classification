import multiprocessing
import numpy as np
import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec
from gensim.models.deprecated.keyedvectors import KeyedVectors
from matplotlib import pyplot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == '__main__':
	model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
	model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')
	embeddings_index = {}
	for w in model_ug_cbow.wv.vocab.keys():
		embeddings_index[w] = np.append(model_ug_cbow.wv[w], model_ug_sg.wv[w])
	print('Found %s word vectors.' % len(embeddings_index))
	print(embeddings_index)
	np.append(model_ug_cbow.wv['công_chúa'], model_ug_sg.wv['công_chúa'])

