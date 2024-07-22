import cupy as cu
from C_Functions import *

class Layer:
    def __init__(self) -> None:
        self.weights = cu.zeros(shape=(input.shape[1], 1))
        self.biases = cu.zeros(shape=(1,))
    def __call__(self, input) -> cu.ndarray:
        return add(cu.matmul(input, self.weights), self.biases)

class Initializer:
    class RandomNormal:
        def __call__(self, inputs, outputs) -> cu.ndarray:
            return cu.random.normal(size=(inputs, outputs))
    class Zeros:
        def __call__(self, inputs) -> cu.ndarray:
            return cu.zeros(inputs)

class Activation:
    class GeLU(Layer):
        def __call__(self, inputs) -> cu.ndarray:
            outputs = cu.empty(inputs.shape)
            gelu((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs
    class Softmax(Layer):
        def __call__(self, inputs) -> cu.ndarray:
            outputs = cu.empty(inputs.shape)
            softmax((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs

class Dense(Layer):
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
        if self.activation != None:
            return self.activation(add(cu.matmul(input, self.weights), self.biases))
        return add(cu.matmul(input, self.weights), self.biases)

class Embedding(Layer):
    ...

class DotProductAttention(Layer):
    def __call__(self, queries, keys, values, d_k, mask=None) -> cu.ndarray:
        scores = div(cu.matmul(queries, keys), cu.sqrt(d_k))
        if mask is not None:
            scores = add(scores, mul(-1e9, mask))
        weights = Activation.Softmax()(scores)
        return cu.matmul(weights, values)

class MultiHeadAttention(Layer):
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

class Normalization(Layer):
    def __call__(self, input) -> cu.ndarray:
        outputs = cu.empty(inputs.shape)
        normalization((inputs[0].shape), (inputs[1].shape), (inputs, inputs.mean(), inputs.var(), outputs))
        return outputs

class Add(Layer):
    def __call__(self, layers) -> cu.ndarray:
        sum = 0
        for layer in layers: sum = add(sum, layer)
        return sum
