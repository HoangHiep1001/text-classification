import glob2
from tqdm import tqdm

train_path = '../../data/data-raw/*/*.txt'

# Hàm đọc file txt
def read_txt(path):
  with open(path, 'r', encoding='utf-8') as f:
    data = f.read()
  return data

# Hàm tạo dữ liệu huấn luyện cho tập train và test
def make_data(path):
  texts = []
  labels = []
  for file_path in tqdm(glob2.glob(train_path)):
    # print(file_path)
    try:
      content = read_txt(file_path)
      print(file_path)
      label = file_path.split('\\')[1]
      texts.append(content)
      labels.append(label)
    except:
      print("error")
      next
  return texts, labels
text_train, label_train = make_data(train_path)
import pickle

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj

# Lưu lại các files
print(label_train)
_save_pkl('../../data/text_train.pkl', text_train)
_save_pkl('../../data/label_train.pkl', label_train)
print('text content:\n', text_train[0])
print('label:\n', label_train[0])
print("done")