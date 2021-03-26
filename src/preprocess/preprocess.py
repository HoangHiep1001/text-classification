import codecs
import os
import re
import pandas as pd
from pyvi import ViTokenizer

from src.preprocess import nlp_utils
from src.utils.unicode import covert_unicode
import csv


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
    text = nlp_utils.chuan_hoa_dau_cau_tieng_viet(text)
    text = nlp_utils.chuan_hoa_dau_tu_tieng_viet(text)
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


if __name__ == '__main__':
    s = []
    s.append('game')
    s.append('du_lich')
    s.append('giao_duc')
    s.append('kinh_doanh')
    s.append('ngan_hang')
    s.append('suc_khoe')
    s.append('the_thao')
    s.append('thoi_su_phap_luat')
    for str1 in s:
        path_in = '../../data/data-raw/' + str1
        path_rs = "../../data/data_process/" + str1
        data = read_file(path_in)
        with codecs.open(path_rs, "w", encoding="utf8") as file:
            for str in data:
                str = text_preprocess(str)
                if len(str) < 200:
                    continue
                file.write(str + "\n")
        print("done " + path_rs)