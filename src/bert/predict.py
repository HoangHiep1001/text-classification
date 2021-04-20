import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.bert.load_data import read_data
from src.bert.test import dataloader_from_text, BERTClassifier

model = BERTClassifier(8)
checkpoint = torch.load('../../model/model_best_valoss.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict_dataloader(dataloader, classes, tokenizer):
    for batch in dataloader:
        batch = tuple(t.to("cpu") for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=None)
            logits = outputs
            logits = logits.detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            print("[PREDICT] {}:{}".format(classes[int(pred_flat)], tokenizer.decode(b_input_ids)))


if __name__ == '__main__':
    classes = ['dien_anh', 'du_lich', 'suc_khoe', 'giao_duc', 'kinh_doanh', 'ngan_hang', 'the_thao',
               'thoi_su_phap_luat']
    train_path = '../../data/data_process/'
    data, label = read_data(path_raw=train_path, classes=classes)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=False)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True)

    train_loader = dataloader_from_text(X_train, y_train, tokenizer=tokenizer, classes=classes, max_len=256,
                                        batch_size=16)
    predict_dataloader(dataloader=train_loader, classes=classes, tokenizer=tokenizer)
