# Chương trình phân loại văn bản sử dụng các mô hình học sâu kết hợp với các mô hình học sẵn
#1. Các công nghệ sử dụng 
- Python 3.6
- Hệ điều hành ubuntu 18.04
- Thư viện gensim, tensorflow, keras, ...
- Google Colab
#2. Chạy chương trình và dự đoán kết quả
* Train model
    - Với mô hình cnn: python model/cnn_model.py
    - Với mô hình lstm: python model/lstm_model.py
    - Với mô hình bi-lstm: python model/bi_lstm_model.py
    - Với mô hình bi-lstm: python bert/bert_model.py
* Chạy chương trình dự đoán phân loại
    - python service/server.py
    - đầu vào: một đoạn văn bản và loại mô hình sử dụng để dự đoán bao gồm
    ['cnn', 'lstm', 'bi-lstm', 'bert']
#3. Dữ liệu sử dụng
* Dữ liệu crawl: https://drive.google.com/drive/folders/1Q7PjejNWwKbeFJcDYfvKtDp6XfB-lCFq?usp=sharing
* Dữ liệu VNTC: https://github.com/duyvuleo/VNTC/tree/master/Data/10Topics/Ver1.1



