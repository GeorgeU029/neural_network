import numpy as np
import struct
from array import array

class MnistDataLoader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        # Store file paths for training images, training labels,
        # test images, and test labels
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Read labels from the file
        with open(labels_filepath, 'rb') as file:
            # Read the magic number and the number of labels
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            # Convert the label bytes into a NumPy array of unsigned bytes
            labels = np.array(array("B", file.read()))

        # Read images from the file
        with open(images_filepath, 'rb') as file:
            # Read the magic number, number of images, number of rows, and columns
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            # Convert the image data into a NumPy array of unsigned bytes
            image_data = np.array(array("B", file.read()))

        # Reshape the flat image data into (size, rows*cols) and normalize pixel values to [0, 1]
        images = image_data.reshape(size, rows * cols) / 255.0

        return images, labels

    def load_data(self):
        # Load training images and labels
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        # Load test images and labels
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# Define file paths for the MNIST dataset
training_images_filepath = 'data_set/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = 'data_set/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = 'data_set/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = 'data_set/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

# Create an instance of the MNIST data loader with the specified file paths
mnist_dataloader = MnistDataLoader(
    training_images_filepath, training_labels_filepath,
    test_images_filepath, test_labels_filepath
)

# Load the data
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Reshape the images into (number of samples, 28*28)
x_train = np.array(x_train).reshape(len(x_train), 28 * 28)
x_test = np.array(x_test).reshape(len(x_test), 28 * 28)

def one_hot_encode(labels, num_classes=10):
    # Convert integer labels into one-hot encoded vectors
    return np.eye(num_classes)[labels]

# One-hot encode the labels for training and testing
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)
