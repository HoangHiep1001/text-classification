from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import regex as re
import numpy as np
from src.preprocess.preprocess import text_preprocessV2, text_preprocess, txtTokenizer
def preProcess(text):
    text = re.sub(r'([^\s\w]|_)+', '', text)
    text = text.lower().strip()
    #print("Tex=",text)
    return text
def predict(inputStr):
    cats = ['1','2','3','4','5','6','7','8']
    texts = []
    sentences = inputStr.split('.')
    sentences = preProcess(sentences)
    texts = texts + sentences
    tokenizer, word_index = txtTokenizer(texts)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X, maxlen=20000)
    model = load_model('../../model/predict_model_#v2.h5')
    prediction = model.predict(X)
    for row in prediction:
        print(row, '=>', cats[row.argmax()])
if __name__ == '__main__':
    pathModel = "../../model/predict_model_#v2.h5"
    model = load_model(pathModel)
    str = 'Ngoài diễn xuất, ca hát, các sao Hàn còn đầu tư vào bất động sản.Hiepej 123'
    tokenizer, word_index = txtTokenizer(str)
    X = tokenizer.texts_to_sequences(str)
    X = pad_sequences(X)
    model = load_model('../../model/predict_model_#v2.h5')
    prediction = model.predict(X)
    print(prediction)
