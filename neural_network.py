import numpy as np
import struct
from array import array

# Dense (Fully-Connected) Layer

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with He initialization and biases to zero
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        # Initialize momentum terms for weights and biases (used in the update)
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

    def forward(self, inputs):
        # Save input for use in backward pass
        self.inputs = inputs
        # Compute the linear transformation
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate=0.01, momentum=0.9):
        # Calculate gradients on weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Calculate gradient on the inputs for backpropagation
        self.dinputs = np.dot(dvalues, self.weights.T)
        # Update momentum terms
        self.weight_momentum = momentum * self.weight_momentum - learning_rate * self.dweights
        self.bias_momentum = momentum * self.bias_momentum - learning_rate * self.dbiases
        # Update weights and biases using the momentum terms
        self.weights += self.weight_momentum
        self.biases += self.bias_momentum


# Leaky ReLU Activation Function

class Activation_LeakyReLU:
    def forward(self, inputs):
        # Store input for backpropagation
        self.inputs = inputs
        # Apply LeakyReLU: if input > 0, return input; else, return 0.1 * input
        self.output = np.where(inputs > 0, inputs, 0.1 * inputs)

    def backward(self, dvalues):
        # Gradient of LeakyReLU: pass gradient unchanged for positive inputs,
        # and multiply by 0.1 for negative inputs
        self.dinputs = np.where(self.inputs > 0, dvalues, 0.1 * dvalues)


# Softmax Activation Function

class Activation_Softmax:
    def forward(self, inputs):
        # Subtract the max value from each input for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the exponentiated values to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues, y_true):
        # Number of samples in the batch
        samples = len(dvalues)
        # If the true labels are one-hot encoded, convert them to class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy dvalues to avoid modifying the original array
        self.dinputs = dvalues.copy()
        # Subtract 1 from the probability of the correct class for each sample
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradients by the number of samples
        self.dinputs /= samples


# Categorical Cross-Entropy Loss

class Loss_CategoricalCrossentropy:
    def calculate(self, y_pred, y_true):
        # Number of samples in the batch
        samples = len(y_pred)
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # If labels are provided as class indices, select the corresponding confidences
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            # If labels are one-hot encoded, sum the confidences across classes
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Return the mean negative log likelihood
        return np.mean(-np.log(correct_confidences))


# Batch Normalization Layer

class BatchNorm:
    def __init__(self, n_neurons, momentum=0.9, epsilon=1e-7):
        # Initialize gamma (scale) and beta (shift) parameters
        self.gamma = np.ones((1, n_neurons))
        self.beta = np.zeros((1, n_neurons))
        # Momentum for updating running mean and variance
        self.momentum = momentum
        # Small epsilon value to avoid division by zero
        self.epsilon = epsilon
        # Initialize running mean and variance for inference
        self.running_mean = np.zeros((1, n_neurons))
        self.running_var = np.ones((1, n_neurons))

    def forward(self, inputs, training=True):
        # Save inputs for use in the backward pass
        self.inputs = inputs
        if training:
            # Compute mean and variance for the current batch
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)
            # Normalize the inputs using the batch statistics
            self.normalized = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            # Scale and shift the normalized inputs
            self.output = self.gamma * self.normalized + self.beta
            # Update running statistics for inference using momentum
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # During inference, use the running mean and variance to normalize
            self.normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * self.normalized + self.beta

    def backward(self, dvalues):
        # Get the number of samples (N) and number of features (D)
        N, D = dvalues.shape
        # Compute gradient with respect to gamma (scale parameter)
        self.dgamma = np.sum(dvalues * self.normalized, axis=0, keepdims=True)
        # Compute gradient with respect to beta (shift parameter)
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient of the loss with respect to the normalized inputs
        dnormalized = dvalues * self.gamma
        # Gradient with respect to the variance
        dvar = np.sum(dnormalized * (self.inputs - self.batch_mean) * -0.5 * (self.batch_var + self.epsilon) ** (-1.5),
                      axis=0, keepdims=True)
        # Gradient with respect to the mean
        dmean = np.sum(dnormalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * (self.inputs - self.batch_mean), axis=0, keepdims=True)
        # Gradient with respect to the inputs
        self.dinputs = dnormalized / np.sqrt(self.batch_var + self.epsilon) + \
                       dvar * 2 * (self.inputs - self.batch_mean) / N + \
                       dmean / N
