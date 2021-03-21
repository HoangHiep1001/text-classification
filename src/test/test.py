import pandas as pd

if __name__ == '__main__':
    path_rs = "../../data/data_process.csv"
    data = pd.read_csv(path_rs, sep=",", encoding='utf-8')
    print(data['content'].head(5))