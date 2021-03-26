import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
from src.preprocess.tranfer_content_to_matrix import comment_embedding
from src.preprocess.word2vec import read_data

if __name__ == '__main__':
    train_data = []
    train_label = []
    content, label = read_data()
    for x in tqdm(content):
        train_data.append(comment_embedding(x))
    train_data = np.array(train_data)
    le = preprocessing.LabelEncoder()
    le.fit(['bat_dong_san','cong_nghe','dien_anh','du_lich','game','giao_duc','kinh_doanh','ngan_hang','suc_khoe','the_thao','thoi_su_phap_thuat'])
    print(label)
    label_encode = le.fit_transform(label)
    print(label_encode)
    for y in tqdm(label_encode):
        label_ = np.zeros(2)
        try:
            label_[int(y)] = 1
        except:
            label_[0] = 1
        train_label.append(label_)
    print(train_label)

# #model
# import numpy as np
# from tensorflow.keras import layers
# from tensorflow import keras
# import tensorflow as tf
#
# sequence_length = 1500
# embedding_size = 250
# num_classes = 3
# filter_sizes = 3
# num_filters = 150
# epochs = 50
# batch_size = 30
# learning_rate = 0.01
# dropout_rate = 0.5
#
# x_train = train_data.reshape(train_data.shape[0], sequence_length, embedding_size, 1).astype('float32')
# y_train = np.array(train_label)
# num_train = 30000
# train_x = x_train[0:num_train]
# test_x = y_train[0:num_train]
# train_y = x_train[num_train:]
# test_y = y_train[num_train:]
# # Define model
# model = keras.Sequential()
# model.add(layers.Convolution2D(num_filters, (filter_sizes, embedding_size),
#                         padding='valid',
#                         input_shape=(sequence_length, embedding_size, 1), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(198, 1)))
# model.add(layers.Dropout(dropout_rate))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(2, activation='softmax'))
# # Train model
# adam = tf.optimizers.Adam()
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
# print(model.summary())
# model.fit(train_x, train_y, validation_split=0.2,shuffle=True,
#                  batch_size=32, epochs=50, verbose=1)
# score = model.evaluate(test_x, test_y, verbose=0)
# print(score)
# model.save('models.h5')