import torch
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

from src.bert.bert_model import BERTClassifier

classes = ['dien_anh', 'du_lich', 'suc_khoe', 'giao_duc', 'kinh_doanh', 'ngan_hang', 'the_thao', 'thoi_su_phap_luat']

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=False)


model = BERTClassifier(len(classes))

path = "../../model/model_bert.pth"
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()


def predict_text(text):
    ids = [tokenizer.encode(text)]
    ids_padded = pad_sequences(ids, maxlen=256, dtype="long", value=0, truncating="post", padding="post")
    mask = [[int(token_id > 0) for token_id in ids_padded[0]]]
    intput_mask = torch.tensor(mask)
    input_ids = torch.tensor(ids_padded)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=intput_mask, labels=None)
        logits = logits.logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        print(pred_flat)
        print("[PREDICT] {}:{}".format(classes[int(pred_flat)], text))
        return pred_flat


if __name__ == '__main__':
    text = 'mexico triệu đàn_ông bệnh bất_lực hiện triệu nam_giới mexico tuổi chứng bất_lực nguyên_nhân chủ_yếu căn_bệnh tim_mạch tiểu_đường chán_nản trầm_uất việc_làm chủ_tịch nghiên_cứu tình_dục mexico những chứng_bệnh mexico gia_tăng bệnh tiểu_đường kết_quả thăm_dò mới_đây nam_giới pháp tây_ban brazil mexico những bệnh bất_lực khủng_hoảng tinh_thần sức_khỏe suy_giảm việc_làm cảnh_báo chứng bất_lực trở_thành thách_thức đối_với y_tế cộng_đồng xã_hội gia_đình ảnh_hưởng trực_tiếp người_bệnh nguy_cơ tan_vỡ hạnh_phúc nhiều vợ_chồng thông_báo y_tế mexico chết bệnh ung_thư chết ung_thư phổi chết ung_thư tử_cung trường_hợp chết ung_thư tuyến tiền_liệt'
    predict_text(text)
