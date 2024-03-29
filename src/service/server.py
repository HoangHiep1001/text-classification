import os
from os import listdir

import numpy as np
from flask import Flask
from flask import request
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from src.bert.predict import predict_text
from src.preprocess.preprocess import text_preprocess

sep = os.sep


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
                    labels.append(folder)
                del all_of_it, sentences
    return content, labels


def read_data_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


app = Flask(__name__)
model_cnn = load_model("../../model/predict_model_cnn.h5")
model_lstm = load_model("../../model/predict_model_bi_lstm.h5")
model_bi_lstm = load_model("../../model/predict_model_bi_lstm.h5")
content, labels = read_data('../../data/data_process')
tokenize = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[]^`{|}~ ', num_words=30000)
tokenize.fit_on_texts(content)
content_sequences = tokenize.texts_to_sequences(content)
pad_train_sequence = pad_sequences(content_sequences, maxlen=500, truncating='post', padding='post')

classes = ['dien_anh', 'du_lich', 'suc_khoe', 'giao_duc', 'kinh_doanh', 'ngan_hang', 'the_thao', 'thoi_su_phap_luat']


@app.route("/classification-text", methods=['POST'])
def text_classification():
    result = ""
    rs = []
    data = request.get_json()
    path = data.get('text', '')
    input_text = read_data_input(path=path)
    input_text = text_preprocess(input_text)
    type_model = data.get('type_model','')
    text = [input_text]
    test_seq = tokenize.texts_to_sequences(text)
    padding_test_seq = pad_sequences(test_seq, maxlen=500, truncating='post', padding='post')
    if type_model == 'cnn':
        rs = model_cnn.predict(padding_test_seq)
    elif type_model == 'lstm':
        rs = model_lstm.predict(padding_test_seq)
    elif type_model == 'bi_lstm':
        rs = model_bi_lstm.predict(padding_test_seq)
    elif type_model == 'bert':
        rs = predict_text(text=input_text)

    if type_model == 'bert':
        print("[PREDICT] {}:{}".format(classes[int(rs)], text))
        result = classes[int(rs)]
    else:
        print('text predict: ', input_text)
        print('label predict: ', classes[np.argmax(rs)])
        result = classes[np.argmax(rs)]
    return result


if __name__ == "__main__":
    app.run()
