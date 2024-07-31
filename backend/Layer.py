import os
if os.environ["USE_CUDA"] == "1":
    import cupy as cu
    if os.environ["USE_C"] == "1":
        from C_Functions import *
else:
    import numpy as cu
from Activation import Softmax

class Initializer:
    class RandomNormal:
        def __call__(self, inputs, outputs) -> cu.ndarray:
            return cu.random.normal(size=(inputs, outputs))
    class Zeros:
        def __call__(self, inputs) -> cu.ndarray:
            return cu.zeros(inputs)

class Dense:
    def __init__(self,
                inputs: int, units: int,
                activation=None,
                kernel_initializer=Initializer.RandomNormal(),
                bias_initializer=Initializer.Zeros(),
                use_bias: bool=True,
                training: bool=True) -> None:
        self.inputs = inputs
        self.units = units
        self.weights = kernel_initializer(inputs, units)
        self.biases = None
        self.activation = activation
        self.optimizer = None
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        if use_bias:
            self.biases = bias_initializer(units)
        if training:
            self.build
    def __call__(self, input) -> cu.ndarray:
        self.in_data = input
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            if self.activation != None:
                return self.activation(linear(self.weights, input, self.biases))
            return linear(self.weights, input, self.biases)
        if self.activation != None:
                return self.activation(self.weights*input+self.biases)
        return self.weights*input+self.biases
    def build(self) -> None:
        self.v_b = self.bias_initializer(self.biases.shape)
        self.m_b = self.bias_initializer(self.biases.shape)
        self.v_b_hat = self.bias_initializer(self.biases.shape)
        self.m_b_hat = self.bias_initializer(self.biases.shape)

        self.v = self.bias_initializer(self.weights.shape)
        self.m = self.bias_initializer(self.weights.shape)
    def set_optimizer(self, optimizer=None) -> None:
        self.optimizer = optimizer
    def backward(self, loss) -> cu.ndarray:
        self.gradient_weights = cu.matmul(self.in_data.transpose(0, 2, 1), loss).sum(axis=0)
        self.gradient_biases = loss.sum(axis=(0, 1))
        out_loss = cu.dot(loss, self.weights.T)
        return out_loss
    def update_weights(self, layer):
        self.weights, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update_weights(self.gradient_weights, self.weights, self.v, self.m, self.v_hat, self.m_hat, layer)
        if self.use_bias:
            self.biases, self.vb, self.mb, self.vb_hat, self.mb_hat  = self.optimizer.update_weights(self.gradient_biases, self.biases, self.vb, self.mb, self.vb_hat, self.mb_hat, layer)
        return layer + 1
    def get_gradients(self):
        return self.gradient_weights, self.gradient_biases
    def set_gradients(self, gradient) -> None:
        self.gradient_weights, self.gradient_biases = gradient

class Embedding:
    def __init__(self,
                in_dim: int, out_dim: int,
                kernel_initializer=Initializer.RandomNormal(),
                bias_initializer=Initializer.Zeros(),
                training: bool=True) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = kernel_initializer(self.in_dim, self.out_dim)
        self.bias_initializer = bias_initializer
        self.optimizer = None
        if training:
            self.build()
    def __call__(self, inputs) -> cu.ndarray:
        self.in_data = inputs
        self.in_length = len(self.in_data[0])
        self.batch_size = len(self.in_data)
        self.in_data = self.encode(self.in_data)
        self.out_data = cu.dot(self.in_data, self.weights)
        return self.out_data
    def build(self) -> None:
        self.v = self.bias_initializer(self.weights.shape)
        self.m = self.bias_initializer(self.weights.shape)
        self.v_hat = self.bias_initializer(self.weights.shape)
        self.m_hat = self.bias_initializer(self.weights.shape)
    def set_optimizer(self, optimizer=None) -> None:
        self.optimizer = optimizer
    def encode(self, labels) -> cu.ndarray:
        labels = labels.astype(cu.int64)
        prepare = self.bias_initializer((labels.size, self.in_dim))
        prepare[cu.arange(labels.size), labels.reshape(1, -1)] = 1
        return prepare.reshape(self.batch_size, self.in_length, self.in_dim)
    def backward(self, loss) -> None:
        self.gradient_weights = cu.matmul(cu.transpose(self.in_data, axes=(0, 2, 1)), loss).sum(axis=0)
    def update_weights(self, layer):
        self.weights, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update_weights(self.gradient_weights, self.weights, self.v, self.m, self.v_hat, self.m_hat, layer)
        return layer + 1
    def get_gradients(self):
        return self.gradient_weights, self.gradient_biases
    def set_gradients(self, gradient) -> None:
        self.gradient_weights, self.gradient_biases = gradient

class DotProductAttention:
    def __call__(self, q, k, v, d_k, mask=None, use_values=True) -> cu.ndarray:
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            scores = div(cu.matmul(q, k), cu.sqrt(d_k))
            if mask is not None:
                scores = add(scores, mul(-1e9, mask))
        else:
            scores = cu.matmul(q, k) / cu.sqrt(d_k)
            if mask is not None:
                scores = scores + -1e9 * mask
        weights = Softmax()(scores)
        if use_values:
            return cu.matmul(weights, v)
        return weights

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

        self.weights = self.attention(self.q, self.k, None, self.d_k, mask=mask, use_values=False)

        out = cu.matmul(self.weights, self.v)
        concat = self.backward_split(out)
        f_o = self.linear_o(concat)

        return f_o
    def set_optimizer(self, optimizer=None):
        self.linear_k.set_optimizer(optimizer)
        self.linear_q.set_optimizer(optimizer)
        self.linear_v.set_optimizer(optimizer)
        self.linear_o.set_optimizer(optimizer)
    def backward(self, loss):
        loss = self.linear_o.backward(loss)
        loss = self.forward_split(loss)
        v_loss = cu.matmul(self.weights.transpose(0, 1, 3, 2), loss)    
        loss = self.attention(loss, self.v.transpose(0, 1, 3, 2), None, self.d_k, use_values=False)
        q_loss = cu.matmul(loss, self.k)
        k_loss = cu.matmul(self.q.transpose(0, 1, 3, 2), loss).transpose(0, 1, 3, 2)
        
        v_loss = self.backward_split(v_loss)
        q_loss = self.backward_split(q_loss)
        k_loss = self.backward_split(k_loss)

        v_loss = self.linear_v(v_loss)
        q_loss = self.linear_q(q_loss)
        k_loss = self.linear_k(k_loss)

        return q_loss, k_loss, v_loss
    def update_weights(self, layer):
        layer = self.linear_k.update_weights(layer)
        layer = self.linear_q.update_weights(layer)
        layer = self.linear_v.update_weights(layer)
        layer = self.linear_o.update_weights(layer)

        return layer

class Normalization:
    def __call__(self, inputs) -> cu.ndarray:
        if os.environ["USE_CUDA"] == 1 and os.environ["USE_C"] == 1:
            outputs = cu.empty(inputs.shape)
            normalization((inputs[0].shape), (inputs[1].shape), (inputs, inputs.mean(), inputs.var(), outputs))
            return outputs
        return (inputs-inputs.mean())/(cu.sqrt(inputs.var()))

class Add:
    def __call__(self, layers) -> cu.ndarray:
        sum = 0
        for layer in layers: sum = add(sum, layer)
        return sum
