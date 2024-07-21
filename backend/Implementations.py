import cupy as cu
from C_Functions import add, sub, mul, div, gelu, softmax, normalization

class Layer:
    def __init__(self):
        self.weights = cu.zeros(shape=(input.shape[1], 1))
        self.biases = cu.zeros(shape=(1,))
        ...
    def __call__(self, input):
        return add(cu.matmul(input, self.weights), self.biases)

class Dense(Layer):
    def __init__(self, inputs, outputs, learning_rate=0.1, activation=None):
        self.learning_rate = learning_rate
        self.weights = mul(cu.random.randn(inputs, outputs), 10e-3)
        self.biases = cu.zeros(outputs)
        self.activation = activation
    def __call__(self, input):
        if self.activation != None:
            return self.activation(add(cu.matmul(input, self.weights), self.biases))
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
    def __call__(self, input):
        return self.gelu(input)
    def backward(self, input, gradient_output):
        gelu_gradient = self.gelu(input)
        return mul(gradient_output, gelu_gradient)

class Softmax(Layer):
    def __init__(self):
        ...
    def softmax(self, x):
        res = cu.empty(x.shape)
        softmax((x[0].shape), (x[1].shape), (x, res))
        return res
    def __call__(self, input):
        return self.softmax(input)
    def backward(self, input, gradient_output):
        softmax_gradient = self.softmax(input)
        return mul(gradient_output, softmax_gradient)

class DotProductAttention(Layer):
    def __init__(self):
        ...
    def __call__(self, queries, keys, values, d_k, mask=None):
        scores = div(cu.matmul(queries, keys), cu.sqrt(d_k))
        if mask is not None:
            scores = add(scores, mul(-1e9, mask))
        weights = Softmax()(scores)
        return cu.matmul(weights, values)

class MultiHeadAttention(Layer):
    def __init__(self, heads, d_k, d_v, d_model):
        self.attention = DotProductAttention()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_q = Dense(d_k, d_k)
        self.W_k = Dense(d_k, d_v)
        self.W_v = Dense(d_v, d_model)
        self.W_o = Dense(d_model, d_model)
    def reshape(self, x, heads, flag=False):
        if flag:
            x = x.reshape(cu.shape(x)[0], cu.shape(x)[1], heads, -1)
            x = x.transpose(0, 2, 1, 3)
        else:
            x = x.transpose(0, 2, 1, 3)
            x = x.reshape(cu.shape(x)[0], cu.shape(x)[1], self.d_k)
        return x
    def __call__(self, queries, keys, values, mask=None):
        q_reshaped = self.reshape(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshape(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape(self.W_v(values), self.heads, True)
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        output = self.reshape(o_reshaped, self.heads, False)
        return self.W_o(output)

class Normalization(Layer):
    def __init__(self):
        ...
    def __call__(self, input):
        res = cu.empty(input.shape)
        normalization((input[0].shape), (input[1].shape), (input, input.mean(), input.var(), res))
        return res

class Add(Layer):
    def __init__(self):
        ...
    def __call__(self, layers):
        return cu.sum(layers) #layers == list
