import pandas as pd

def tranfer_csv(path):
    data = pd.read_csv(path, sep="\t", encoding='utf-8')
    data.to_csv('data.csv',sep=';', header=False)

if __name__ == '__main__':
    tranfer_csv("../../data/data-test.csv")