import gensim.models.keyedvectors as word2vec
import numpy as np


pathModel = '../../model/word.model'

model_embedding = word2vec.KeyedVectors.load(pathModel)

word_labels = []
max_seq = 1500
embedding_size = 250

for word in model_embedding.vocab.keys():
    word_labels.append(word)


def comment_embedding(content):
    matrix = np.zeros((max_seq, embedding_size))
    words = content.split()
    lencmt = len(words)

    for i in range(max_seq):
        indexword = i % lencmt
        if (max_seq - i < lencmt):
            break
        if (words[indexword] in word_labels):
            matrix[i] = model_embedding[words[indexword]]
    matrix = np.array(matrix)
    return matrix

if __name__ == '__main__':
    print(comment_embedding('tổng công_ty cổ xuất nhập_khẩu xây_dựng việt_nam thông_báo_cáo_bạch niêm_yết phụ_lục sau bat_dong_san'))