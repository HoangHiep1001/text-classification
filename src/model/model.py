import numpy as np
from tqdm import tqdm

from src.preprocess.tranfer_content_to_matrix import comment_embedding
from src.preprocess.word2vec import read_data

if __name__ == '__main__':
    train_data = []
    label_data = []
    content, label = read_data()
    for x in tqdm(content):
        train_data.append(comment_embedding(x))
    train_data = np.array(train_data)

    for y in tqdm(label):
        label_ = np.zeros(4)
        try:
            label_[int(y)] = 1
        except:
            label_[0] = 1
        label_data.append(label_)
    print(train_data)

#model
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

sequence_length = 200
embedding_size = 128
num_classes = 3
filter_sizes = 3
num_filters = 150
epochs = 50
batch_size = 30
learning_rate = 0.01
dropout_rate = 0.5

x_train = train_data.reshape(train_data.shape[0], sequence_length, embedding_size, 1).astype('float32')
y_train = np.array(label_data)

# Define model
model = keras.Sequential()
model.add(layers.Convolution2D(num_filters, (filter_sizes, embedding_size),
                        padding='valid',
                        input_shape=(sequence_length, embedding_size, 1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(198, 1)))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
# Train model
adam = tf.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
print(model.summary())
model.fit(x = x_train[:4000], y = y_train[:4000], batch_size = batch_size, verbose=1, epochs=epochs, validation_data=(x_train[:5997], y_train[:5997]))

model.save('models.h5')