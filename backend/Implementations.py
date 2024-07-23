import cupy as cu
from C_Functions import *

#Basic layer class
class Layer:
    def __init__(self) -> None:
        self.weights = cu.zeros(shape=(input.shape[1], 1))
        self.biases = cu.zeros(shape=(1,))
    def __call__(self, input) -> cu.ndarray:
        return linear(self.weights, input, self.biases)

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
            return self.activation(linear(self.weights, input, self.biases))
        return linear(self.weights, input, self.biases)
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
    def __call__(self, queries, keys, d_k, mask=None) -> cu.ndarray:
        scores = div(cu.matmul(queries, keys), cu.sqrt(d_k))
        if mask is not None:
            scores = add(scores, mul(-1e9, mask))
        weights = Activation.Softmax()(scores)
        return weights

#MultiHeadAttention layer class
class MultiHeadAttention:
    def __init__(self, heads: int, d_mdl: int) -> None:
        self.heads = heads
        self.d_mdl = d_mdl

        self.attention = DotProductAttention()

        self.d_k = self.d_mdl // heads
        self.d_q = self.d_mdl // heads
        self.d_v = self.d_mdl // heads

        self.linear_k = Dense(self.d_mdl, self.d_k * heads)
        self.linear_q = Dense(self.d_mdl, self.d_q * heads)
        self.linear_v = Dense(self.d_mdl, self.d_v * heads)
        self.linear_o = Dense(self.d_mdl, self.d_v * heads)
    def forward_split(self, inputs) -> cu.ndarray:
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1, self.heads, self.d_k).transpose(0, 2, 1, 3)
    def backward_split(self, inputs) -> cu.ndarray:
        batch_size = inputs.shape[0]
        return inputs.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.heads * self.d_k)
    def __call__(self, q, k, v, mask=None) -> cu.ndarray:
        self.len_k = k.shape[1]
        self.len_q = q.shape[1]
        self.len_v = v.shape[1]

        f_k = self.linear_k(k)
        f_q = self.linear_q(q)
        f_v = self.linear_v(v)

        self.k = self.forward_split(f_k)
        self.q = self.forward_split(f_q)
        self.v = self.forward_split(f_v)

        self.weights = self.attention(self.q, self.k, self.d_k, mask=mask)

        out = cu.matmul(self.weights, self.v)
        concat = self.backward_split(out)
        f_o = self.linear_o(concat)

        return f_o
    def backward(self, loss):
        loss = self.linear_o.backward(loss)
        loss = self.forward_split(loss)
        v_loss = cu.matmul(self.scores.transpose(0, 1, 3, 2), loss)    
        loss = self.attention(loss, self.v.transpose(0, 1, 3, 2), self.d_k)
        q_loss = cu.matmul(loss, self.k)
        k_loss = cu.matmul(self.q.transpose(0, 1, 3, 2), loss).transpose(0, 1, 3, 2)
        
        v_loss = self.backward_split(v_loss)
        q_loss = self.backward_split(q_loss)
        k_loss = self.backward_split(k_loss)

        v_loss = self.linear_v(v_loss)
        q_loss = self.linear_q(q_loss)
        k_loss = self.linear_k(k_loss)

        return q_loss, k_loss, v_loss

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
