from keras.models import load_model
import numpy as np
from src.preprocess.tranfer_content_to_matrix import comment_embedding
from src.preprocess.preprocess import text_preprocess
model_sentiment = load_model("models.h5")

text = "Cao Thái Sơn - ngoài việc làm ca sĩ thì còn là một nhà đầu tư bất động sản (BĐS) cực kì mạnh tay. Tại một chương trình truyền hình, trước thắc mắc có bao nhiêu căn nhà, Cao Thái Sơn từng  khẳng định: Thật ra hỏi bao nhiêu căn nhà như hỏi mình có bao nhiêu bài hát, buộc phải đếm. Quả thật, biệt thự của chàng ca sĩ trải dài tại nhiều nơi, căn nào cũng rất hoành tráng nhìn mà mê.Thậm chí, Trấn Thành còn từng bật mí về cơ ngơi của Cao Thái Sơn là: Thằng em của em nó chỉ có mỗi quận 10 căn nhà. Tính sơ sơ từ đây qua Mỹ, mỗi thành phố 1 căn nhà thôi, đếm đi, tôi không biết làm toán"
text1 = "Khác với dòng Galaxy S21 Series được làm mỏng đi, hộp của Galaxy A 2021 lần này lại vẫn theo phong cách hơi dày cộp một chút. Hãy cùng tìm hiểu xem bên trong Samsung đã phóng khoáng tặng thêm gì cho người dùng"

text = text_preprocess(text)

maxtrix_embedding = np.expand_dims(comment_embedding(text), axis=0)
maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)

result = model_sentiment.predict(maxtrix_embedding)
print("Label predict: ", result)