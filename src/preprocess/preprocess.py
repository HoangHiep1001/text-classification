import codecs
import re
import pandas as pd
from pyvi import ViTokenizer

from src.preprocess import nlp_utils
from src.utils.unicode import covert_unicode
import csv

def read_file(filePath):
    list = []
    with codecs.open(filePath,"r",encoding="utf8") as file:
        a = file.readlines()
        for i in a :
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
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopword(text):
    filename = '../../data/stopwords.csv'
    data = pd.read_csv(filename, sep="\t", encoding='utf-8')
    list_stopwords = data['stopwords']
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
        text2 = ' '.join(pre_text)
    return text2
def write_data_process(path,text):
    with codecs.open(path,"w",encoding="utf8") as file:
        file.write(text+"\n")
if __name__ == '__main__':
    path_in = '../../data/data.txt'
    path_rs = "../../data/data_process.csv"
    data = read_file(path_in)
    df = []
    for text in data:
        try:
            content = text.split("==")[0];
            label = text.split("==")[1];
            a = text_preprocess(content)
            b = text_preprocess(label)
            df.append(a + "," + b)
        except:
            print("No data")
    with open(path_rs, 'w', newline='',encoding='utf-8') as csvfile:
        fieldnames = ['content', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for str in df:
            ct = str.split(",")[0]
            lb = str.split(",")[1]
            writer.writerow({'content':''+ct, 'category': ''+lb})