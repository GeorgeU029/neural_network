import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate=0.01, momentum=0.9):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

        self.weight_momentum = momentum * self.weight_momentum - learning_rate * self.dweights
        self.bias_momentum = momentum * self.bias_momentum - learning_rate * self.dbiases

        self.weights += self.weight_momentum
        self.biases += self.bias_momentum


class Activation_LeakyReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, 0.1 * inputs)  

    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs > 0, dvalues, 0.1 * dvalues) 

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:  
            y_true = np.argmax(y_true, axis=1)  

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples  




class Loss_CategoricalCrossentropy:
    def calculate(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return np.mean(-np.log(correct_confidences))
import numpy as np

class BatchNorm:
    def __init__(self, n_neurons, momentum=0.9, epsilon=1e-7):
        self.gamma = np.ones((1, n_neurons))
        self.beta = np.zeros((1, n_neurons))
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, n_neurons))
        self.running_var = np.ones((1, n_neurons))

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)
            self.normalized = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            self.output = self.gamma * self.normalized + self.beta
            # Update running averages for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            self.normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * self.normalized + self.beta

    def backward(self, dvalues):
        N, D = dvalues.shape
        
        self.dgamma = np.sum(dvalues * self.normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)
        
        dnormalized = dvalues * self.gamma
        
        dvar = np.sum(dnormalized * (self.inputs - self.batch_mean) * -0.5 * (self.batch_var + self.epsilon) ** (-1.5), axis=0, keepdims=True)
        
        dmean = np.sum(dnormalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * (self.inputs - self.batch_mean), axis=0, keepdims=True)
        
        self.dinputs = dnormalized / np.sqrt(self.batch_var + self.epsilon) + \
                       dvar * 2 * (self.inputs - self.batch_mean) / N + \
                       dmean / N


