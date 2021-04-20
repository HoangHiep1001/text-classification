import codecs
import os
import pickle
import re
from os import listdir

import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from pyvi import ViTokenizer

from src.preprocess.chuan_hoa_dau import chuan_hoa_dau_cau_tieng_viet
from src.preprocess.nlp_utils import chuan_hoa_dau_tu_tieng_viet
from src.utils.unicode import covert_unicode


def txtTokenizer(texts):
    tokenizer = Tokenizer(num_words=20000)
    # fit the tokenizer on our text
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    return tokenizer, word_index

def read_file(filePath):
    list = []
    with codecs.open(filePath, "r", encoding="utf8") as file:
        a = file.readlines()
        for i in a:
            list.append(i)
    return list


def removeHtml(text):
    return re.sub(r'<[^>]*>', '', text)


def unicodeConvert(text):
    return covert_unicode(text)


def text_preprocess(text):
    text = unicodeConvert(text)
    text = removeHtml(text)
    text = chuan_hoa_dau_cau_tieng_viet(text)
    text = chuan_hoa_dau_tu_tieng_viet(text)
    text = remove_stopword(text)
    text = ViTokenizer.tokenize(text)
    text = text.lower()
    # xóa các ký tự không cần thiết
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', '', text).strip()
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
        if word not in list_stopwords:
            pre_text.append(word)
        text2 = ' '.join(pre_text)
    return text2

sep = os.sep


def preProcess(sentences):

    text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in sentences if sentence!='']
    text = [sentence.lower().strip().split() for sentence in text]
    #print("Tex=",text)
    return text

def text_preprocessV2(sentences):
    text = [unicodeConvert(sentence) for sentence in sentences if sentence!='']
    # text = [chuan_hoa_dau_cau_tieng_viet(sentence) for sentence in text if sentence!='']
    text = [chuan_hoa_dau_tu_tieng_viet(sentence) for sentence in text if sentence!='']
    text = [remove_stopword(sentence) for sentence in text if sentence!='']
    text = [ViTokenizer.tokenize(sentence) for sentence in text if sentence!='']
    text = [sentence.lower() for sentence in text if sentence!='']
    # xóa các ký tự không cần thiết
    text = [re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', sentence) for sentence in text if sentence!='']
    text = [re.sub(r'\b\w{1,3}\b', '', sentence).strip() for sentence in text if sentence!='']
    text = [re.sub(r'\d+', ' ', sentence).strip() for sentence in text if sentence!='']
    text = [re.sub(r'\s+', ' ', sentence).strip() for sentence in text if sentence!='']
    text = [sentence.lower().strip().split() for sentence in text]
    return text

def loadData(data_folder):

    texts = []
    labels = []
    #
    for folder in listdir(data_folder):
        for file in listdir(data_folder + sep + folder):
            with open(data_folder + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                print("read file: "+data_folder + sep + folder + sep + file)
                all_of_it = f.read()
                sentences = all_of_it.split('\n')
                sentences = text_preprocessV2(sentences)
                print(sentences)
                texts = texts + sentences
                label = [folder for _ in sentences]
                labels = labels + label
                del all_of_it, sentences
    return texts, labels

def dump_data(data_folder):
    texts, labels = loadData(data_folder)
    tokenizer, word_index = txtTokenizer(texts)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X)

    # prepare the labels
    y = pd.get_dummies(labels)
    file = open("../../data/data.pkl", 'wb')
    pickle.dump([X,y, texts],file)
    file.close()

if __name__ == '__main__':
    test_text = ['video quảng_bình cánh đồng hướng_dương thu_hút du_khách video clip trước cánh đồng hướng_dương rộng ha trồng cao_nguyên mục_đích thức_ăn nuôi công_ty cao_lương loài mặt_trời cảnh_sắc tuyệt_vời nở_rộ khiến trở_thành thu_hút đông_du_khách hằng triệu mặt_trời đồng_loạt nở_rộ khoe rộng_lớn cao_nguyên huyện nghĩa nghệ_an cánh đồng mặt_trời nở_rộ trên mảnh miền trung khiến trầm_trồ đắm_đuối vùng_đất_đỏ bazan vàng mặt_trời trải chân_trời vàng mênh_mang hướng_dương vàng thu_hút du_khách tuổi nhiều chụp trang_phục nhau áo_dài trang_phục bình_thường năng_động phù_hợp dáng loài cánh đồng hướng_dương đặc_biệt trục đường minh đoạn chạy huyện nghĩa mở_cửa du_khách tháng những bông đầu_tiên khoe đông_du_khách thiên_đường hướng_dương nhất việt_nam những khoảnh_khắc vàng rực_rỡ thiên_đường chục những ngày thêm cánh đồng hướng_dương vàng rực_rỡ nở_rộ thu_hút đông_du_khách sinh_thái diễn huyện diễn châu nghệ_an những cánh đồng hướng_dương khoe hứa_hẹn ý_nghĩa cơ_hội vùng_đất tỉnh nghệ quảng_bá hình_ảnh quê_hương phát_triển loại_hình dịch_vụ địa_phương trục đường minh chạy huyện nghĩa cánh đồng hướng_dương trở_thành thuận_tiện đông_du_khách tỉnh nghệ_an triệu mặt_trời đồng_loạt nở_rộ khoe rộng_lớn cánh đồng mặt_trời trở_thành thu_hút đông_du_khách những ngày sinh_thái diễn huyện diễn châu nghệ_an những bông hướng_dương màu_sắc rực_rỡ dương_lịch phục_vụ du_khách những luống trồng hàng_lối thẳng_tắp gieo nhau tuần nở_rộ kéo_dài tháng những cánh mặt_trời nở_rộ thời_kỳ nhất hướng_dương_tỏa vàng rực_rỡ chính nhất nhiều gia_đình thăm những khoảnh_khắc vui_tươi những mặt_trời']
    text = 'video quảng_bình cánh đồng hướng_dương thu_hút du_khách video clip trước cánh đồng hướng_dương rộng ha trồng cao_nguyên mục_đích thức_ăn nuôi công_ty cao_lương loài mặt_trời cảnh_sắc tuyệt_vời nở_rộ khiến trở_thành thu_hút đông_du_khách hằng triệu mặt_trời đồng_loạt nở_rộ khoe rộng_lớn cao_nguyên huyện nghĩa nghệ_an cánh đồng mặt_trời nở_rộ trên mảnh miền trung khiến trầm_trồ đắm_đuối vùng_đất_đỏ bazan vàng mặt_trời trải chân_trời vàng mênh_mang hướng_dương vàng thu_hút du_khách tuổi nhiều chụp trang_phục nhau áo_dài trang_phục bình_thường năng_động phù_hợp dáng loài cánh đồng hướng_dương đặc_biệt trục đường minh đoạn chạy huyện nghĩa mở_cửa du_khách tháng những bông đầu_tiên khoe đông_du_khách thiên_đường hướng_dương nhất việt_nam những khoảnh_khắc vàng rực_rỡ thiên_đường chục những ngày thêm cánh đồng hướng_dương vàng rực_rỡ nở_rộ thu_hút đông_du_khách sinh_thái diễn huyện diễn châu nghệ_an những cánh đồng hướng_dương khoe hứa_hẹn ý_nghĩa cơ_hội vùng_đất tỉnh nghệ quảng_bá hình_ảnh quê_hương phát_triển loại_hình dịch_vụ địa_phương trục đường minh chạy huyện nghĩa cánh đồng hướng_dương trở_thành thuận_tiện đông_du_khách tỉnh nghệ_an triệu mặt_trời đồng_loạt nở_rộ khoe rộng_lớn cánh đồng mặt_trời trở_thành thu_hút đông_du_khách những ngày sinh_thái diễn huyện diễn châu nghệ_an những bông hướng_dương màu_sắc rực_rỡ dương_lịch phục_vụ du_khách những luống trồng hàng_lối thẳng_tắp gieo nhau tuần nở_rộ kéo_dài tháng những cánh mặt_trời nở_rộ thời_kỳ nhất hướng_dương_tỏa vàng rực_rỡ chính nhất nhiều gia_đình thăm những khoảnh_khắc vui_tươi những mặt_trời'
    list_text = []
    list_text.append(text)
    print(test_text)