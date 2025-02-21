from neural_network import *
from read import *
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model parameters from the file
data = np.load("model.npz")

# Initialize the network layers along with their BatchNorm and activation functions

# Layer 1: Input (784 features) to 256 neurons
dense1 = Layer_Dense(784, 256)
batch1 = BatchNorm(256)
activation1 = Activation_LeakyReLU()

# Layer 2: 256 neurons to 128 neurons
dense2 = Layer_Dense(256, 128)
batch2 = BatchNorm(128)
activation2 = Activation_LeakyReLU()

# Layer 3: 128 neurons to 64 neurons
dense3 = Layer_Dense(128, 64)
batch3 = BatchNorm(64)
activation3 = Activation_LeakyReLU()

# Layer 4 (Output): 64 neurons to 10 neurons
dense4 = Layer_Dense(64, 10)
activation4 = Activation_Softmax()

# Load Dense layer parameters (weights and biases) from the saved file
dense1.weights, dense1.biases = data["w1"], data["b1"]
dense2.weights, dense2.biases = data["w2"], data["b2"]
dense3.weights, dense3.biases = data["w3"], data["b3"]
dense4.weights, dense4.biases = data["w4"], data["b4"]

# Load BatchNorm parameters for each layer from the saved file
batch1.gamma, batch1.beta = data["bn1_gamma"], data["bn1_beta"]
batch1.running_mean, batch1.running_var = data["bn1_running_mean"], data["bn1_running_var"]

batch2.gamma, batch2.beta = data["bn2_gamma"], data["bn2_beta"]
batch2.running_mean, batch2.running_var = data["bn2_running_mean"], data["bn2_running_var"]

batch3.gamma, batch3.beta = data["bn3_gamma"], data["bn3_beta"]
batch3.running_mean, batch3.running_var = data["bn3_running_mean"], data["bn3_running_var"]

print("Model loaded successfully")

# ---------------------------
# Forward pass on the entire test dataset
# ---------------------------
dense1.forward(x_test)
batch1.forward(dense1.output, training=False)  # Use inference mode for BatchNorm
activation1.forward(batch1.output)

dense2.forward(activation1.output)
batch2.forward(dense2.output, training=False)
activation2.forward(batch2.output)

dense3.forward(activation2.output)
batch3.forward(dense3.output, training=False)
activation3.forward(batch3.output)

dense4.forward(activation3.output)
activation4.forward(dense4.output)  # Apply Softmax to obtain probabilities

# Compute predictions and test accuracy
test_predictions = np.argmax(activation4.output, axis=1)
true_labels = np.argmax(y_test, axis=1)
test_accuracy = np.mean(test_predictions == true_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ---------------------------
# Forward pass on a single random test sample
# ---------------------------
index = np.random.randint(0, len(x_test))
sample_image = x_test[index]

# Reshape the sample image and perform a forward pass
dense1.forward(sample_image.reshape(1, -1))
batch1.forward(dense1.output, training=False)
activation1.forward(batch1.output)

dense2.forward(activation1.output)
batch2.forward(dense2.output, training=False)
activation2.forward(batch2.output)

dense3.forward(activation2.output)
batch3.forward(dense3.output, training=False)
activation3.forward(batch3.output)

dense4.forward(activation3.output)
activation4.forward(dense4.output)

# Get the predicted label for the sample
predicted_label = np.argmax(activation4.output)
print(f"Single Sample Prediction: {predicted_label}, True: {np.argmax(y_test[index])}")

# Display the test image along with its predicted and true label
plt.imshow(sample_image.reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_label}, True: {np.argmax(y_test[index])}")
plt.axis("off")
plt.show()
