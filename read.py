import numpy as np
import struct
from array import array

class MnistDataLoader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):

        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = np.array(array("B", file.read()))  

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = np.array(array("B", file.read()))  

        images = image_data.reshape(size, rows * cols) / 255.0

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


training_images_filepath = 'data_set/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = 'data_set/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = 'data_set/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = 'data_set/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

mnist_dataloader = MnistDataLoader(
    training_images_filepath, training_labels_filepath,
    test_images_filepath, test_labels_filepath
)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.array(x_train).reshape(len(x_train), 28 * 28)
x_test = np.array(x_test).reshape(len(x_test), 28 * 28)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

