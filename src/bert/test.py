from os import listdir

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
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


def make_mask(batch_ids):
    batch_mask = []
    for ids in batch_ids:
        mask = [int(token_id > 0) for token_id in ids]
        batch_mask.append(mask)
    return torch.tensor(batch_mask)


class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, num_labels, bert_model, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()
        if bert_model is not None:
            self.roberta = bert_model
        else:
            self.roberta = RobertaModel.from_pretrained("vinai/phobert-base", local_files_only=False)
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, num_labels)

    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)
        return x


class BERTClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        bert_classifier_config = RobertaConfig.from_pretrained(
            "../../vinai/phobert-base/config.json",
            from_tf=False,
            num_labels=num_labels,
            output_hidden_states=False,
        )
        print("LOAD BERT PRETRAIN MODEL")
        self.bert_classifier = RobertaForSequenceClassification.from_pretrained(
            "../../vinai/phobert-base/model.bin",
            config=bert_classifier_config
        )

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert_classifier(input_ids=input_ids,
                                      token_type_ids=None,
                                      attention_mask=attention_mask,
                                      labels=labels
                                      )
        return output


class ClassifierTrainner():
    def __init__(self, bert_model, train_dataloader, valid_dataloader, epochs=10, cuda_device="cpu", save_dir=None):

        self.device = torch.device('cuda:{}'.format(cuda_device))
        self.model = bert_model
        if save_dir is not None and os.path.exists(save_dir):
            print("Load weight from file:{}".format(save_dir))
            self.save_dir = save_dir
            epcho_checkpoint_path = glob.glob("{}/model_epoch*".format(self.save_dir))
            if len(epcho_checkpoint_path) == 0:
                print("No checkpoint found in: {}\nCheck save_dir...".format(self.save_dir))
            else:
                self.load_checkpoint(epcho_checkpoint_path)
                print("Restore weight successful from: {}".format(epcho_checkpoint_path))
        else:
            self.save_dir = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            os.makedirs(self.save_dir)
            print("Training new model, save to: {}".format(self.save_dir))

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs
        # self.batch_size = batch_size

    def save_checkpoint(self, save_path):
        state_dict = {'model_state_dict': self.model.state_dict()}
        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

    def load_checkpoint(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)
        print(f'Model restored from <== {load_path}')
        self.model.load_state_dict(state_dict['model_state_dict'])

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        F1_score = f1_score(pred_flat, labels_flat, average='macro')
        return accuracy_score(pred_flat, labels_flat), F1_score

    def train_classifier(self):
        self.model.to(self.device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

        for epoch_i in range(0, self.epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            total_loss = 0
            self.model.train()
            train_accuracy = 0
            nb_train_steps = 0
            train_f1 = 0
            best_valid_loss = 999999
            best_eval_accuracy = 0
            for step, batch in enumerate(self.train_dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = make_mask(batch[0]).to(self.device)
                b_labels = batch[1].to(self.device)

                self.model.zero_grad()
                outputs = self.model(b_input_ids,
                                     attention_mask=b_input_mask,
                                     labels=b_labels
                                     )
                loss = outputs[0]
                total_loss += loss.item()

                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.cpu().numpy()
                tmp_train_accuracy, tmp_train_f1 = self.flat_accuracy(logits, label_ids)
                train_accuracy += tmp_train_accuracy
                train_f1 += tmp_train_f1
                nb_train_steps += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                if step % 100 == 0:
                    print(
                        "[TRAIN] Epoch {}/{} | Batch {}/{} | Train Loss={} | Train Acc={}".format(epoch_i, self.epochs,
                                                                                                  step, len(
                                self.train_dataloader), loss.item(), tmp_train_accuracy))

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(" Train Accuracy: {0:.4f}".format(train_accuracy / nb_train_steps))
            print(" Train F1 score: {0:.4f}".format(train_f1 / nb_train_steps))
            print(" Train Loss: {0:.4f}".format(avg_train_loss))

            print("Running Validation...")
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            eval_f1 = 0

            for batch in self.valid_dataloader:
                b_input_mask = make_mask(batch[0]).to(self.device)
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_labels = batch
                with torch.no_grad():
                    outputs = self.model(b_input_ids,
                                         attention_mask=b_input_mask,
                                         labels=b_labels
                                         )
                    tmp_eval_loss, logits = outputs[0], outputs[1]
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.cpu().numpy()
                    tmp_eval_accuracy, tmp_eval_f1 = self.flat_accuracy(logits, label_ids)
                    eval_accuracy += tmp_eval_accuracy
                    eval_loss += tmp_eval_loss
                    eval_f1 += tmp_eval_f1
                    nb_eval_steps += 1

            print(" Valid Loss: {0:.4f}".format(eval_loss / nb_eval_steps))
            print(" Valid Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
            print(" Valid F1 score: {0:.4f}".format(eval_f1 / nb_eval_steps))

            if best_valid_loss > eval_loss:
                best_valid_loss = eval_loss
                best_valid_loss_path = "{}/model_best_valoss.pt".format(self.save_dir)
                self.save_checkpoint(best_valid_loss_path)
            if best_eval_accuracy > eval_accuracy:
                best_eval_accuracy = eval_accuracy
                best_eval_accuracy_path = "{}/model_best_valacc.pt".format(self.save_dir)
                self.save_checkpoint(best_eval_accuracy_path)

            epoch_i_path = "{}/model_epoch{}.pt".format(self.save_dir, epoch_i)
            self.save_checkpoint(epoch_i_path)
            os.remove("{}/model_epoch{}.pt".format(self.save_dir, epoch_i - 1))

        print("Training complete!")

    def predict_dataloader(self, dataloader, classes, tokenizer):
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                     attention_mask=b_input_mask,
                                     labels=None
                                     )
                logits = outputs
                logits = logits.detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()
                print("[PREDICT] {}:{}".format(classes[int(pred_flat)], tokenizer.decode(b_input_ids)))

    def predict_text(self, text, classes, tokenizer, max_len=256):
        ids = tokenizer.encode(text)
        ids_padded = pad_sequences(ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
        mask = [int(token_id > 0) for token_id in ids_padded]
        input_ids = torch.tensor(ids_padded)
        intput_mask = torch.tensor(mask)
        with torch.no_grad():
            logits = self.model(input_ids,
                                attention_mask=intput_mask,
                                labels=None
                                )
            logits = logits.detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            print("[PREDICT] {}:{}".format(classes[int(pred_flat)], text))


if __name__ == '__main__':
    classes = ['dien_anh', 'du_lich']
    MAX_LEN = 256
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=False)

    train_path = '../../data/data_process/'
    data, label = read_data(path_raw=train_path, classes=classes)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True)

    train_loader = dataloader_from_text(X_train, y_train, tokenizer=tokenizer, max_len=MAX_LEN,
                                        batch_size=16)
    val_loader = dataloader_from_text(X_test, y_test, tokenizer=tokenizer, max_len=256, batch_size=16)
    bert_classifier_model = BERTClassifier(len(classes))
    # train model
    bert_classifier_trainer = ClassifierTrainner(bert_model=bert_classifier_model, train_dataloader=train_loader,
                                                 valid_dataloader=val_loader, epochs=10, cuda_device="0")
    bert_classifier_trainer.train_classifier()
    print('===train done====')
def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])