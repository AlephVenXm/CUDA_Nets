import cupy as cu
from C_Functions import *

#Basic layer class
class Layer:
    def __init__(self) -> None:
        self.weights = cu.zeros(shape=(input.shape[1], 1))
        self.biases = cu.zeros(shape=(1,))
    def __call__(self, input) -> cu.ndarray:
        return add(cu.matmul(input, self.weights), self.biases)

#Kernel and bias initializer class
class Initializer:
    class RandomNormal:
        def __call__(self, inputs, outputs) -> cu.ndarray:
            return cu.random.normal(size=(inputs, outputs))
    class Zeros:
        def __call__(self, inputs) -> cu.ndarray:
            return cu.zeros(inputs)

#Activation class
class Activation:
    class ReLU:
        def __call__(self, inputs) -> cu.ndarray:
            outputs = cu.empty(inputs.shape)
            relu((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs
        def backward(self, gradient) -> cu.ndarray:
            ...
            return self(gradient)
    class GeLU:
        def __call__(self, inputs) -> cu.ndarray:
            outputs = cu.empty(inputs.shape)
            gelu((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs
        def backward(self, gradient) -> cu.ndarray:
            ...
            return self(gradient)
    class Softmax:
        def __call__(self, inputs) -> cu.ndarray:
            outputs = cu.empty(inputs.shape)
            softmax((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs
        def backward(self, gradient) -> cu.ndarray:
            ...
            return self(gradient)
    class Sigmoid:
        def __call__(self, inputs, param=1.0) -> cu.ndarray:
            outputs = cu.empty(inputs.shape)
            sigmoid((inputs.shape[0],), (inputs.shape[1],), (inputs, param, outputs))
            return outputs
        def backward(self, gradient) -> cu.ndarray:
            ...
            return self(gradient)

#Standard dense layer class
class Dense:
    def __init__(self,
                inputs: int, outputs: int,
                activation=None,
                kernel_initializer=Initializer.RandomNormal(),
                bias_initializer=Initializer.Zeros(),
                use_bias: bool=True) -> None:
        self.weights = kernel_initializer(inputs, outputs)
        self.biases = None
        self.activation = activation
        if use_bias:
            self.biases = bias_initializer(outputs)
    def __call__(self, input) -> cu.ndarray:
        self.in_data = input
        if self.activation != None:
            return self.activation(add(cu.matmul(input, self.weights), self.biases))
        return add(cu.matmul(input, self.weights), self.biases)
    def backward(self, loss) -> cu.ndarray:
        self.gradient_weights = cu.matmul(self.in_data.transpose(0, 2, 1), loss).sum(axis=0)
        self.gradient_bias = loss.sum(axis=(0, 1))
        out_loss = cu.dot(loss, self.w.T)
        return out_loss
    def update_weights(self, layer):
        ...

#Embedding layer class
class Embedding:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = Initializer.RandomNormal()(self.in_dim, self.out_dim)
        self.v = Initializer.Zeros()(self.weights.shape)
        self.m = Initializer.Zeros()(self.weights.shape)
    def __call__(self, inputs) -> cu.ndarray:
        self.in_data = inputs
        self.in_length = len(self.in_data[0])
        self.batch_size = len(self.in_data)
        self.in_data = self.encode(self.in_data)
        self.out_data = cu.dot(self.in_data, self.weights)
        return self.out_data
    def encode(self, labels) -> cu.ndarray:
        labels = labels.astype(cu.int64)
        prepare = Initializer.Zeros()((labels.size, self.in_dim))
        prepare[cu.arange(labels.size), labels.reshape(1, -1)] = 1
        return prepare.reshape(self.batch_size, self.in_length, self.in_dim)
    def backward(self, loss) -> None:
        self.gradient_weights = cu.matmul(cu.transpose(self.in_data, axes=(0, 2, 1)), loss).sum(axis=0)
    def update_weights(self, layer):
        ...

#DotProductAttention layer class
class DotProductAttention:
    def __call__(self, queries, keys, values, d_k, mask=None) -> cu.ndarray:
        scores = div(cu.matmul(queries, keys), cu.sqrt(d_k))
        if mask is not None:
            scores = add(scores, mul(-1e9, mask))
        weights = Activation.Softmax()(scores)
        return cu.matmul(weights, values)

#MultiHeadAttention layer class
class MultiHeadAttention:
    def __init__(self, heads, d_k, d_v, d_model) -> None:
        self.attention = DotProductAttention()
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_q = Dense(d_k, d_k)
        self.W_k = Dense(d_k, d_v)
        self.W_v = Dense(d_v, d_model)
        self.W_o = Dense(d_model, d_model)
    def reshape(self, x, heads, flag=False) -> cu.ndarray:
        if flag:
            x = x.reshape(cu.shape(x)[0], cu.shape(x)[1], heads, -1)
            x = x.transpose(0, 2, 1, 3)
        else:
            x = x.transpose(0, 2, 1, 3)
            x = x.reshape(cu.shape(x)[0], cu.shape(x)[1], self.d_k)
        return x
    def __call__(self, queries, keys, values, mask=None) -> cu.ndarray:
        q_reshaped = self.reshape(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshape(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshape(self.W_v(values), self.heads, True)
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        output = self.reshape(o_reshaped, self.heads, False)
        return self.W_o(output)

#Layer normalization class
class Normalization:
    def __call__(self, inputs) -> cu.ndarray:
        outputs = cu.empty(inputs.shape)
        normalization((inputs[0].shape), (inputs[1].shape), (inputs, inputs.mean(), inputs.var(), outputs))
        return outputs

#Add for residual connection of layers
class Add:
    def __call__(self, layers) -> cu.ndarray:
        sum = 0
        for layer in layers: sum = add(sum, layer)
        return sum
