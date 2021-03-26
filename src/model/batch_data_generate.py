from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import keras

class DataGenerator(Sequence):
    def __init__(self,
                 paths,
                 labels,
                 batch_size=32,
                 dim=(224, 224),
                 n_channels=3,
                 n_classes=4,
                 shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.paths = paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.img_indexes = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temps):
        X = np.empty((self.batch_size, *self.dim))
        y = []
        for i, ID in enumerate(list_IDs_temps):
            X[i,] = self.img_paths[ID]
            X = (X / 255).astype('float32')
            y.append(self.labels[ID])
        X = X[:, :, :, np.newaxis]
        return X, keras.utils.to_categorical(y, num_classes=11)