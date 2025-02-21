from neural_network import *   # Import all neural network components (layers, activations, etc.)
from read import *             # Import data reading utilities
import numpy as np

# Set training hyperparameters
epochs = 50                  # Number of training epochs
learning_rate = 0.1          # Learning rate for weight updates
loss_function = Loss_CategoricalCrossentropy()  # Initialize the categorical cross-entropy loss

# Initialize the network layers

# First layer: Input (784 features) to 256 neurons
dense1 = Layer_Dense(784, 256)
batch1 = BatchNorm(256)      # Batch normalization for the first layer
activation1 = Activation_LeakyReLU()  # Leaky ReLU activation for the first layer

# Second layer: 256 neurons to 128 neurons
dense2 = Layer_Dense(256, 128)
batch2 = BatchNorm(128)      # Batch normalization for the second layer
activation2 = Activation_LeakyReLU()  # Leaky ReLU activation for the second layer

# Third layer: 128 neurons to 64 neurons
dense3 = Layer_Dense(128, 64)
batch3 = BatchNorm(64)       # Batch normalization for the third layer
activation3 = Activation_LeakyReLU()  # Leaky ReLU activation for the third layer

# Fourth (output) layer: 64 neurons to 10 neurons (one per class)
dense4 = Layer_Dense(64, 10)
activation4 = Activation_Softmax()  # Softmax activation for output probabilities

# Training loop
for epoch in range(epochs):

    # ----- Forward Pass -----
    # Layer 1
    dense1.forward(x_train)                        # Compute output of dense layer 1
    batch1.forward(dense1.output, training=True)   # Apply batch normalization (training mode)
    activation1.forward(batch1.output)             # Apply LeakyReLU activation

    # Layer 2
    dense2.forward(activation1.output)             # Compute output of dense layer 2
    batch2.forward(dense2.output, training=True)   # Apply batch normalization (training mode)
    activation2.forward(batch2.output)             # Apply LeakyReLU activation

    # Layer 3
    dense3.forward(activation2.output)             # Compute output of dense layer 3
    batch3.forward(dense3.output, training=True)   # Apply batch normalization (training mode)
    activation3.forward(batch3.output)             # Apply LeakyReLU activation

    # Layer 4 (Output)
    dense4.forward(activation3.output)             # Compute output of dense layer 4
    activation4.forward(dense4.output)             # Apply Softmax to obtain class probabilities

    # Compute loss for the current epoch
    loss = loss_function.calculate(activation4.output, y_train)

    # Compute training accuracy
    predictions = np.argmax(activation4.output, axis=1)
    true_labels = np.argmax(y_train, axis=1)
    accuracy = np.mean(predictions == true_labels)

    # ----- Backward Pass -----
    # Backpropagation through the output layer
    activation4.backward(activation4.output, y_train)  # Compute gradient of softmax with loss
    dense4.backward(activation4.dinputs, learning_rate)  # Update weights of dense layer 4

    # Backpropagation through Layer 3
    activation3.backward(dense4.dinputs)              # Backward pass through activation layer 3
    batch3.backward(activation3.dinputs)              # Backward pass through batch normalization layer 3
    dense3.backward(batch3.dinputs, learning_rate)    # Update weights of dense layer 3

    # Backpropagation through Layer 2
    activation2.backward(dense3.dinputs)              # Backward pass through activation layer 2
    batch2.backward(activation2.dinputs)              # Backward pass through batch normalization layer 2
    dense2.backward(batch2.dinputs, learning_rate)    # Update weights of dense layer 2

    # Backpropagation through Layer 1
    activation1.backward(dense2.dinputs)              # Backward pass through activation layer 1
    batch1.backward(activation1.dinputs)              # Backward pass through batch normalization layer 1
    dense1.backward(batch1.dinputs, learning_rate)    # Update weights of dense layer 1

    # Print the loss and accuracy for the current epoch
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

# Save model parameters (dense layer weights and biases, and batch normalization parameters)
np.savez("model.npz",
         w1=dense1.weights, b1=dense1.biases,
         w2=dense2.weights, b2=dense2.biases,
         w3=dense3.weights, b3=dense3.biases,
         w4=dense4.weights, b4=dense4.biases,
         bn1_gamma=batch1.gamma, bn1_beta=batch1.beta,
         bn1_running_mean=batch1.running_mean, bn1_running_var=batch1.running_var,
         bn2_gamma=batch2.gamma, bn2_beta=batch2.beta,
         bn2_running_mean=batch2.running_mean, bn2_running_var=batch2.running_var,
         bn3_gamma=batch3.gamma, bn3_beta=batch3.beta,
         bn3_running_mean=batch3.running_mean, bn3_running_var=batch3.running_var)

print("Model saved successfully!")
