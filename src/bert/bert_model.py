import glob
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, AdamW, RobertaModel, AutoTokenizer, BertConfig

from src.bert.load_data import make_mask, read_data, dataloader_from_text

sep = os.sep


class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, num_labels, bert_model, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()
        if bert_model is not None:
            self.roberta = bert_model
        else:
            self.roberta = RobertaModel.from_pretrained("vinai/phobert-base", local_files_only=False)
        self.drop1 = torch.nn.Dropout(dropout_rate)
        self.lstm = torch.nn.LSTM(768, 64, batch_first=True, bidirectional=True)
        self.norm = torch.nn.LayerNorm(64)
        self.drop2 = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(64, num_labels)

    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop1(x)
        x = self.lstm(x)
        x = self.norm(x)
        x = torch.nn.Tanh()(x)
        x = self.drop2(x)
        x = self.linear(x)
        return x


class PhoBertClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(PhoBertClassifier, self).__init__()
        bert_classifier_config = BertConfig.from_pretrained(
            "../../vinai/phobert-base/config.json",
            from_tf=False,
            num_labels=num_labels,
            output_hidden_states=False,
        )
        print("====load bert model from trained====")
        # using roberta classification
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


class Train():
    def __init__(self, bert_model, train_dataloader, valid_dataloader, epochs=10, cuda_device="cpu"):

        self.device = torch.device('cuda:{}'.format(cuda_device))
        self.model = bert_model
        self.save_dir = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        os.makedirs(self.save_dir)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs

    def save_checkpoint(self, save_path):
        state_dict = {'model_state_dict': self.model.state_dict()}
        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

    def load_checkpoint(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)
        print(f'Model load from <== {load_path}')
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
            print('Training model!!!')

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
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
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
                if step % 150 == 0:
                    print(
                        "[train] epoch {}/{} | batch {}/{} | train_loss={} | train_acc={}".format(epoch_i, self.epochs,
                                                                                                  step,
                                                                                                  len(
                                                                                                      self.train_dataloader),
                                                                                                  loss.item(),
                                                                                                  tmp_train_accuracy))

            avg_train_loss = total_loss / len(self.train_dataloader)
            print("train_acc: {0:.3f}".format(train_accuracy / nb_train_steps))
            print("train_F1_score: {0:.3f}".format(train_f1 / nb_train_steps))
            print("train_loss: {0:.3f}".format(avg_train_loss))

            print("Validation Model ...")
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            eval_f1 = 0

            for batch in self.valid_dataloader:
                val_mask = make_mask(batch[0]).to(self.device)
                batch = tuple(t.to(self.device) for t in batch)
                val_ids, val_labels = batch
                with torch.no_grad():
                    outputs = self.model(val_ids, attention_mask=val_mask, labels=val_labels)
                    tmp_eval_loss, logits = outputs[0], outputs[1]
                    logits = logits.detach().cpu().numpy()
                    label_ids = val_labels.cpu().numpy()
                    tmp_eval_accuracy, tmp_eval_f1 = self.flat_accuracy(logits, label_ids)
                    eval_accuracy += tmp_eval_accuracy
                    eval_loss += tmp_eval_loss
                    eval_f1 += tmp_eval_f1
                    nb_eval_steps += 1

            print("val_loss: {0:.3f}".format(eval_loss / nb_eval_steps))
            print("val_acc: {0:.3f}".format(eval_accuracy / nb_eval_steps))
            print("val_f1_score: {0:.3f}".format(eval_f1 / nb_eval_steps))

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


if __name__ == '__main__':
    classes = ['dien_anh', 'du_lich', 'suc_khoe', 'giao_duc', 'kinh_doanh', 'ngan_hang', 'the_thao',
               'thoi_su_phap_luat']
    MAX_LEN = 256
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=False)

    train_path = '../../data/data_process/'
    data, label = read_data(path_raw=train_path, classes=classes)

    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.2, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)
    train_loader = dataloader_from_text(X_train, y_train, tokenizer=tokenizer, max_len=MAX_LEN,
                                        batch_size=16)
    val_loader = dataloader_from_text(X_val, y_val, tokenizer=tokenizer, max_len=256, batch_size=16)
    bert_classifier_model = PhoBertClassifier(len(classes))
    # train model
    bert_classifier_trainer = Train(bert_model=bert_classifier_model, train_dataloader=train_loader,
                                    valid_dataloader=val_loader, epochs=10, cuda_device="0")
    bert_classifier_trainer.train_classifier()
    print('===train done====')
