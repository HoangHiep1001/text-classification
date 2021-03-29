from keras.models import load_model
import numpy as np
from src.preprocess.tranfer_content_to_matrix import comment_embedding
from src.preprocess.preprocess import text_preprocess
model_sentiment = load_model("models.h5")

str = "Cao Thái Sơn - ngoài việc làm ca sĩ thì còn là một nhà đầu tư bất động sản (BĐS) cực kì mạnh tay. Tại một chương trình truyền hình, trước thắc mắc có bao nhiêu căn nhà, Cao Thái Sơn từng  khẳng định: Thật ra hỏi bao nhiêu căn nhà như hỏi mình có bao nhiêu bài hát, buộc phải đếm. Quả thật, biệt thự của chàng ca sĩ trải dài tại nhiều nơi, căn nào cũng rất hoành tráng nhìn mà mê.Thậm chí, Trấn Thành còn từng bật mí về cơ ngơi của Cao Thái Sơn là: Thằng em của em nó chỉ có mỗi quận 10 căn nhà. Tính sơ sơ từ đây qua Mỹ, mỗi thành phố 1 căn nhà thôi, đếm đi, tôi không biết làm toán"
str1 = 'Khi mà năm 2020 gần kết thúc, Twitter đã quyết định hỏi người dùng về việc miêu tả 366 ngày sắp qua của năm 2020 bằng một từ. Bài đăng này nhận được rất nhiều sự chú ý của cộng đồng mạng, và đặc biệt là Microsoft, Messenger... cũng đã tham gia vào màn trả lời này với câu đáp trả cực gắt'
text = text_preprocess(str)

maxtrix_embedding = np.expand_dims(comment_embedding(text), axis=0)
maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)

result = model_sentiment.predict(maxtrix_embedding)
print("Label predict: ", result)