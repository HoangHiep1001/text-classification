from os import listdir
import os
import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm

sep = os.sep


def make_mask(batch_ids):
    batch_mask = []
    for ids in batch_ids:
        mask = [int(token_id > 0) for token_id in ids]
        batch_mask.append(mask)
    return torch.tensor(batch_mask)


def read_data(path_raw, classes=[]):
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


def dataloader_from_text(texts=[], labels=[], tokenizer=None, max_len=256, batch_size=16, infer=False):
    print(texts[0])
    print(labels[0])
    # text to id
    ids = []
    for text in tqdm(texts):
        encoded_sent = tokenizer.encode(text)
        ids.append(encoded_sent)
    del texts
    # padding sentences 256 length
    ids_padded = pad_sequences(ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    del ids
    print(ids_padded)

    print("CONVERT TO TORCH TENSOR")
    ids_inputs = torch.tensor(ids_padded)
    del ids_padded

    if not infer:
        labels = torch.tensor(labels)

    print("CREATE DATALOADER")
    if infer:
        input_data = TensorDataset(ids_inputs)
    else:
        input_data = TensorDataset(ids_inputs, labels)
    input_sampler = SequentialSampler(input_data)

    data_loader = DataLoader(input_data, sampler=input_sampler, batch_size=batch_size)

    print("len data_loader:", len(data_loader))
    print("LOAD DATA ALL DONE")
    return data_loader
