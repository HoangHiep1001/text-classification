import codecs
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
    text = re.sub(r'\d+', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopword(text):
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
    path_in = '../../data/data-raw/du_lich'
    path_rs = "../../data/data_process/du_lich"
    data = read_file(path_in)
    with codecs.open(path_rs,"w",encoding="utf8") as file:
        for str in data:
            str = text_preprocess(str)
            file.write(str+"\n")
