import gensim
from gensim.models import KeyedVectors

if __name__ == '__main__':
    # word_vectors = KeyedVectors.load_word2vec_format('../../data/data_dump/word_model.save',binary=True)
    word_model = gensim.models.Word2Vec.load("../../model/word_model_v2.save")
    print(word_model.wv.similar_by_word('tình_hình'))