import gensim.models.keyedvectors as word2vec
import numpy as np
from tqdm import tqdm

model_embedding = word2vec.KeyedVectors.load('./word.model')

words_label = []
max_seq = 200
embedding_size = 200

for word in model_embedding.vocab.keys():
    words_label.append(word)


def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    lencmt = len(words)

    for i in range(max_seq):
        indexword = i % lencmt
        if (max_seq - i < lencmt):
            break
        if (words[indexword] in words_label):
            matrix[i] = model_embedding[words[indexword]]
    matrix = np.array(matrix)
    return matrix
train_data = []
label_data = []

for x in tqdm(pre_reviews):
    train_data.append(comment_embedding(x))
train_data = np.array(train_data)

for y in tqdm(labels):
    label_ = np.zeros(3)
    try:
        label_[int(y)] = 1
    except:
        label_[0] = 1
    label_data.append(label_)
if __name__ == '__main__':
    print(comment_embedding())