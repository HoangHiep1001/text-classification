from os import listdir

from vncorenlp import VnCoreNLP
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
from tqdm import tqdm
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW, RobertaTokenizer, RobertaTokenizerFast, \
    RobertaModel, AutoTokenizer
from datetime import datetime
import glob
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


def dataloader_from_text(text_file=None, tokenizer=None, classes=[], savetodisk=None, segment=False,
                         max_len=256, batch_size=16, infer=False):
    ids_padded, masks, labels = [], [], []
    texts,labels = read_data('../../data/test_data/')
    print("TEXT TO IDS")
    ids = []
    for text in tqdm(texts):
        encoded_sent = tokenizer.encode(text)
        print(encoded_sent)
        ids.append(encoded_sent)

    del texts
    # print("PADDING IDS")
    ids_padded = pad_sequences(ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    del ids
    # print("CREATE MASK")
    # for sent in tqdm(ids_padded):
    #     masks.append(make_mask(sent))
    print(ids_padded)
    if savetodisk != None and not infer:
        with open(savetodisk, 'wb') as f:
            pickle.dump(ids_padded, f)
            # pickle.dump(masks, f)
            pickle.dump(labels, f)
        print("SAVED IDS DATA TO DISK")

    print("CONVERT TO TORCH TENSOR")
    ids_inputs = torch.tensor(ids_padded)
    del ids_padded
    # masks = torch.tensor(masks)
    if not infer:
        labels = torch.tensor(labels)

    print("CREATE DATALOADER")
    if infer:
        # input_data = TensorDataset(ids_inputs, masks)
        input_data = TensorDataset(ids_inputs)
    else:
        input_data = TensorDataset(ids_inputs, labels)
        # input_data = TensorDataset(ids_inputs, masks, labels)
    input_sampler = SequentialSampler(input_data)
    dataloader = DataLoader(input_data, sampler=input_sampler, batch_size=batch_size)

    print("len dataloader:", len(dataloader))
    print("LOAD DATA ALL DONE")
    return dataloader


if __name__ == '__main__':
    classes = ['__label__sống_trẻ', '__label__thời_sự', '__label__công_nghệ', '__label__sức_khỏe', '__label__giáo_dục',
               '__label__xe_360', '__label__thời_trang', '__label__du_lịch', '__label__âm_nhạc', '__label__xuất_bản',
               '__label__nhịp_sống', '__label__kinh_doanh', '__label__pháp_luật', '__label__ẩm_thực',
               '__label__thế_giới', '__label__thể_thao', '__label__giải_trí', '__label__phim_ảnh']
    test_path = 'test.txt'

    MAX_LEN = 256
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=False)

    train_dataloader = dataloader_from_text(tokenizer=tokenizer, classes=classes, savetodisk=None,
                                            max_len=MAX_LEN, batch_size=16)
    print(train_dataloader)
