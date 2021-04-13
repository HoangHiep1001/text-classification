from os import listdir
import os

import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
sep = os.sep()


def read_data(path_file):
    texts = []
    labels = []
    #
    for folder in listdir(path_file):
        for file in listdir(path_file + sep + folder):
            with open(path_file + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                print("read file: " + path_file + sep + folder + sep + file)
                all_of_it = f.read()
                sentences = all_of_it.split('\n')
                texts = texts + sentences
                label = [folder for _ in sentences]
                labels = labels + label
                del all_of_it, sentences
    return texts, labels


def text2id(tokenizer):
    texts, label = read_data("")
    print("TEXT TO IDS")
    ids = []
    for text in tqdm(texts):
        encoded_sent = tokenizer.encode(text)
        ids.append(encoded_sent)
    del texts
    ids_padded = pad_sequences(ids, maxlen=256, dtype="long", value=0, truncating="post", padding="post")
    del ids
    print("CONVERT TO TORCH TENSOR")
    ids_inputs = torch.tensor(ids_padded)
    del ids_padded

    labels = torch.tensor(label)

    print("CREATE DATALOADER")
    input_data = TensorDataset(ids_inputs, labels)
    input_sampler = SequentialSampler(input_data)
    dataloader = DataLoader(input_data, sampler=input_sampler, batch_size=16)

    print("len dataloader:", len(dataloader))
    print("LOAD DATA ALL DONE")
    return dataloader


if __name__ == '__main__':
    text2id()
