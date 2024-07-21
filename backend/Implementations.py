import cupy as cu
from C_Functions import add, sub, mul, div, gelu, softmax

class Layer:
    def __init__(self):
        self.weights = cu.zeros(shape=(input.shape[1], 1))
        self.biases = cu.zeros(shape=(1,))
        ...
    def forward(self, input):
        output = add(cu.matmul(input, self.weights), self.biases)
        return output

class Dense(Layer):
    def __init__(self, inputs, outputs, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = mul(cu.random.randn(inputs, outputs), 10e-3)
        self.biases = cu.zeros(outputs)
    def forward(self, input):
        return add(cu.matmul(input, self.weights), self.biases)
    def backward(self, input, gradient_output):
        gradient_input = cu.dot(gradient_output, cu.transpose(self.weights))
        gradient_weights = cu.transpose(cu.dot(cu.transpose(gradient_output), input))
        gradient_biases = cu.sum(gradient_output, axis=0)
        self.weights = sub(self.weights, mul(self.learning_rate, gradient_weights))
        self.biases = sub(self.biases, mul(self.learning_rate, gradient_biases))
        return gradient_input

class GeLU(Layer):
    def __init__(self):
        ...
    def gelu(self, x):
        res = cu.empty(x.shape)
        gelu((x[0].shape), (x[1].shape), (x, res))
        return res
    def forward(self, input):
        return self.gelu(input)
    def backward(self, input, gradient_output):
        gelu_gradient = self.gelu(input)
        return mul(gradient_output, gelu_gradient)

class Softmax(Layer):
    def __init__(self):
        ...
    def softmax(self, x, axis=-1):
        res = cu.empty(x.shape)
        softmax((x[0].shape), (x[1].shape), (x, res))
        return res
    def forward(self, input):
        return self.softmax(input)
    def backward(self, input, gradient_output):
        softmax_gradient = self.softmax(input)
        return mul(gradient_output, softmax_gradient)
