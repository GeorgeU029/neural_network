from neural_network import *
from read import *
import numpy as np

epochs = 50               
learning_rate = 0.1     
loss_function = Loss_CategoricalCrossentropy()

dense1 = Layer_Dense(784, 256)
batch1 = BatchNorm(256) 
activation1 = Activation_LeakyReLU()

dense2 = Layer_Dense(256, 128)
batch2 = BatchNorm(128)  
activation2 = Activation_LeakyReLU()

dense3 = Layer_Dense(128, 64)
batch3 = BatchNorm(64)  
activation3 = Activation_LeakyReLU()

dense4 = Layer_Dense(64, 10)
activation4 = Activation_Softmax()

for epoch in range(epochs):

    dense1.forward(x_train)
    batch1.forward(dense1.output, training=True)
    activation1.forward(batch1.output)

    dense2.forward(activation1.output)
    batch2.forward(dense2.output, training=True)
    activation2.forward(batch2.output)

    dense3.forward(activation2.output)
    batch3.forward(dense3.output, training=True)
    activation3.forward(batch3.output)

    dense4.forward(activation3.output)
    activation4.forward(dense4.output)

    loss = loss_function.calculate(activation4.output, y_train)

    predictions = np.argmax(activation4.output, axis=1)
    true_labels = np.argmax(y_train, axis=1)
    accuracy = np.mean(predictions == true_labels)

    activation4.backward(activation4.output, y_train)
    dense4.backward(activation4.dinputs, learning_rate)

    activation3.backward(dense4.dinputs)
    batch3.backward(activation3.dinputs)
    dense3.backward(batch3.dinputs, learning_rate)

    activation2.backward(dense3.dinputs)
    batch2.backward(activation2.dinputs)
    dense2.backward(batch2.dinputs, learning_rate)

    activation1.backward(dense2.dinputs)
    batch1.backward(activation1.dinputs)
    dense1.backward(batch1.dinputs, learning_rate)

    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')


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
