import pickle
from os import listdir
import os
import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm

from src.preprocess.preprocess import text_preprocess

sep = os.sep


def read_txt(path):
    with open(path, 'r', encoding='utf-16') as f:
        data = f.read()
    return data


def read_data(path_raw, classes=[]):
    content = []
    labels = []
    for folder in listdir(path_raw):
        print('read folder: ', folder)
        for file in listdir(path_raw + sep + folder):
            with open(path_raw + sep + folder + sep + file, 'r', encoding="utf-16") as f:
                all_of_it = f.read()
                sentences = text_preprocess(all_of_it)
                content.append(sentences)
                labels.append(classes.index(folder))
                del all_of_it, sentences
    return content, labels


classes = ['Chinh tri Xa hoi', 'Doi song', 'Khoa hoc', 'Kinh doanh', 'Phap luat', 'Suc khoe', 'The gioi', 'The thao',
           'Van hoa', 'Vi tinh']

train_path = '/home/hiephm/PycharmProjects/VNTC/Data/10Topics/Ver1.1/test'
data, label = read_data(path_raw=train_path, classes=classes)
file = open("../../data/test.pkl", 'wb')
pickle.dump([data, label], file)
file.close()