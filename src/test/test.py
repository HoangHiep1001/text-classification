import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
sep = os.sep



def read_data(path_process):
    list_length = []
    for folder in listdir(path_process):
        for file in listdir(path_process + sep + folder):
            with open(path_process + sep + folder + sep + file, 'r', encoding="utf-8") as f:
                print("read file: " + path_process + sep + folder + sep + file)
                all_of_it = f.read()
                sentences = all_of_it.split('\n')
                for str in sentences:
                    list_length.append(len(str.split()))
    return list_length


if __name__ == '__main__':
    list_lenth = read_data("../../data/data_process")
    avg_length = sum(list_lenth) / len(list_lenth)
    print(avg_length)
    plt.plot(list_lenth)
    plt.show()