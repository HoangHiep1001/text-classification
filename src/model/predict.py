import pickle

from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from src.preprocess.preprocess import text_preprocess, txtTokenizer


if __name__ == '__main__':
    model_sentiment = load_model("../../model/predict_model_v2.h5")

    str = "Cao Thái Sơn - ngoài việc làm ca sĩ thì còn là một nhà đầu tư bất động sản (BĐS) cực kì mạnh tay. Tại một chương trình truyền hình, trước thắc mắc có bao nhiêu căn nhà, Cao Thái Sơn từng  khẳng định: Thật ra hỏi bao nhiêu căn nhà như hỏi mình có bao nhiêu bài hát, buộc phải đếm. Quả thật, biệt thự của chàng ca sĩ trải dài tại nhiều nơi, căn nào cũng rất hoành tráng nhìn mà mê.Thậm chí, Trấn Thành còn từng bật mí về cơ ngơi của Cao Thái Sơn là: Thằng em của em nó chỉ có mỗi quận 10 căn nhà. Tính sơ sơ từ đây qua Mỹ, mỗi thành phố 1 căn nhà thôi, đếm đi, tôi không biết làm toán"
    str1 = 'Báo Sức Khỏe và Đời Sống – Cơ quan ngôn luận của Bộ Y tế, cung cấp những thông tin chuẩn xác, tin cậy về y tế, chăm sóc sức khoẻ và các vấn đề nóng khác của xã hội. '

    # #test
    file = open("../../data/data_dump/data.pkl", 'rb')
    X, y, texts = pickle.load(file)
    print(X[0])
    text = text_preprocess(str1)
    print(text)
    tokenizer, word_index = txtTokenizer(text)
    text_seq = tokenizer.texts_to_sequences(text)
    # print(text_seq)
    # X = pad_sequences(text_seq,maxlen=6147,truncating='post',padding='post')
    result = model_sentiment.predict(X[0])
    print("Label predict: ", result)