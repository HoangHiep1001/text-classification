import codecs
import os
import pickle
import re
from os import listdir

import gensim
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from pyvi import ViTokenizer

from src.preprocess import nlp_utils
from src.utils.unicode import covert_unicode
import csv


def txtTokenizer(texts):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[]^`{|}~ ', num_words=30000, oov_token='OOV')
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    return tokenizer, word_index


def removeHtml(text):
    return re.sub(r'<[^>]*>', '', text)


def read_file(filePath):
    list = []
    with codecs.open(filePath, "r", encoding="utf8") as file:
        a = file.readlines()
        for i in a:
            list.append(i)
    return list


def unicodeConvert(text):
    return covert_unicode(text)


def text_preprocess(text):
    text = unicodeConvert(text)
    text = nlp_utils.chuan_hoa_dau_cau_tieng_viet(text)
    text = nlp_utils.chuan_hoa_dau_tu_tieng_viet(text)
    text = ViTokenizer.tokenize(text)
    text = text.lower()
    text = remove_stopword(text)
    # xóa các ký tự không cần thiết
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', text)
    text = re.sub(r'\b\w{1,3}\b', '', text).strip()
    text = re.sub(r'\d+', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopword(text):
    global text2
    filename = '../../data/stopwords.txt'
    list_stopwords = []
    with codecs.open(filename, "r", encoding="utf8") as file:
        a = file.readlines()
        for i in a:
            list_stopwords.append(i.strip())
    pre_text = []
    words = text.split()
    for word in words:
        if (word not in list_stopwords) and (word.__contains__('_') or len(word) < 6):
            pre_text.append(word)
        text2 = ' '.join(pre_text)
    return text2


sep = os.sep


def preProcess(sentences):
    text = [re.sub(r'([^\s\w]|)+', '', sentence) for sentence in sentences if sentence != '']
    text = [sentence.lower().strip().split() for sentence in text]
    return text


def text_preprocessV2(sentences):
    text = [unicodeConvert(sentence) for sentence in sentences if sentence != '']
    text = [nlp_utils.chuan_hoa_dau_cau_tieng_viet(sentence) for sentence in text if sentence != '']
    text = [nlp_utils.chuan_hoa_dau_tu_tieng_viet(sentence) for sentence in text if sentence != '']
    text = [ViTokenizer.tokenize(sentence) for sentence in text if sentence != '']
    text = [sentence.lower() for sentence in text if sentence != '']
    text = [remove_stopword(sentence) for sentence in text if sentence != '']
    # xóa các ký tự không cần thiết
    text = [re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', sentence) for
            sentence in text if sentence != '']
    # text = [re.sub(r'\b\w{1,3}\b', '', sentence).strip() for sentence in text if sentence != '']
    text = [re.sub(r'\d+', ' ', sentence).strip() for sentence in text if sentence != '']
    text = [re.sub(r'\s+', ' ', sentence).strip() for sentence in text if sentence != '']
    text = [sentence.lower().strip().split() for sentence in text]
    return text


def loadData(data_folder):
    texts = []
    labels = []
    #
    for folder in listdir(data_folder):
        for file in listdir(data_folder + sep + folder):
            with open(data_folder + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                print("read file: " + data_folder + sep + folder + sep + file)
                all_of_it = f.read()
                sentences = all_of_it.split('\n')
                sentences = preProcess(sentences)
                texts = texts + sentences
                label = [folder for _ in sentences]
                labels = labels + label
                del all_of_it, sentences
    return texts, labels


classes = ['dien_anh', 'du_lich', 'suc_khoe', 'giao_duc', 'kinh_doanh', 'ngan_hang', 'the_thao', 'thoi_su_phap_luat']


def read_data(path_raw):
    content = []
    labels = []
    for folder in listdir(path_raw):
        for file in listdir(path_raw + sep + folder):
            with open(path_raw + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                print("read file: " + path_raw + sep + folder + sep + file)
                all_of_it = f.read()
                sentences = all_of_it.split('\n')
                for str in sentences:
                    content.append(str)
                for _ in sentences:
                    labels.append(classes.index(folder))
                del all_of_it, sentences
    return content, labels


def dump_data(data_folder):
    texts, labels = loadData(data_folder)
    tokenizer, word_index = txtTokenizer(texts)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X)
    y = pd.get_dummies(labels)
    file = open("../../data/data_dump/data.pkl", 'wb')
    pickle.dump([X, y, texts], file)
    file.close()


if __name__ == '__main__':
    s = []
    # content,label = loadData('../../data/data_process/')
    # word_model = gensim.models.Word2Vec(content, size=300, min_count=1, iter=10)
    # word_model.save("../../data/word_model.save")
    # word_model.wv.save_word2vec_format('../../data/test_w2v.txt', binary=False)
    s.append('dien_anh')
    s.append('du_lich')
    s.append('giao_duc')
    s.append('kinh_doanh')
    s.append('ngan_hang')
    s.append('suc_khoe')
    s.append('the_thao')
    s.append('thoi_su_phap_luat')
    for str1 in s:
        path_in = '../../data/data-raw/' + str1 + sep + str1 + '.txt'
        path_rs = "../../data/data_process/" + str1 + sep + str1 + '.txt'
        data = read_file(path_in)
        with codecs.open(path_rs, "w", encoding="utf-8") as file:
            for str in data:
                str = text_preprocess(str)
                if len(str.split()) > 20:
                    file.write(str + "\n")
        print("done " + path_rs)
