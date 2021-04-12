import gensim
from gensim.models import KeyedVectors

if __name__ == '__main__':
    word_model = KeyedVectors.load_word2vec_format('../../model/baomoi.model.bin',binary=True)
    # word_model = gensim.models.Word2Vec.load("../../data/baomoi.model.bin",binary=True)
    print(word_model.wv.vocab)
    print(word_model.wv.most_similar('hộ_tống_chim'))