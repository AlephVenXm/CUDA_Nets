import os
if os.environ["USE_CUDA"] == "1":
    import cupy as cu
    if os.environ["USE_C"] == "1":
        from C_Functions import *
else:
    import numpy as cu

class ReLU:
    def __call__(self, inputs) -> cu.ndarray:
        self.inputs = inputs
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            outputs = cu.empty(inputs.shape)
            relu((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs
        return cu.maximum(0, inputs)
    def backward(self, gradient) -> cu.ndarray:
        f = self(self.inputs)
        #if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
        #    outputs = cu.empty(f.shape)
        #    relu_backward((f.shape[0],), (f.shape[1],), (f, outputs))
        #    return outputs
        if os.environ["USE_CUDA"] == "1":
            return mul(gradient, (mul(f, (1.0 - f))))
        return gradient * (f * (1.0 - f))
    
class GeLU:
    def __call__(self, inputs) -> cu.ndarray:
        self.inputs = inputs
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            outputs = cu.empty(inputs.shape)
            gelu((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs
        return 0.5*inputs*(1+cu.tanh(cu.sqrt(2/cu.pi)*(inputs+0.044715*cu.power(inputs, 3))))
    def backward(self, gradient) -> cu.ndarray:
        inputs = self.inputs
        #if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
        #    outputs = cu.empty(inputs.shape)
        #    gelu_backward((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
        #    return outputs
        sech = lambda y: 2 / (cu.exp(y) + cu.exp(-y))
        return gradient * (0.5 * cu.tanh(0.0356774 * cu.power(inputs, 3) + 0.797885 * inputs)
            + (0.0535161 * cu.power(inputs, 3) + 0.398942 * inputs
            * cu.power(sech(0.0356774 * cu.power(inputs, 3) + 0.797885 * inputs), 2) + 0.5))
    
class Softmax:
    def __call__(self, inputs) -> cu.ndarray:
        self.inputs = inputs
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            outputs = cu.empty(inputs.shape)
            softmax((inputs.shape[0],), (inputs.shape[1],), (inputs, outputs))
            return outputs
        numerator = cu.exp(inputs - cu.max(inputs, axis=-1, keepdims=True))
        return numerator / cu.sum(numerator, axis=-1, keepdims=True)
    def backward(self, gradient) -> cu.ndarray:
        f = self(self.inputs)
        #if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
        #    outputs = cu.empty(f.shape)
        #    softmax_backward((f.shape[0],), (f.shape[1],), (f, outputs))
        #    return outputs
        soft = f[..., cu.newaxis] * cu.tile(cu.identity(f.shape[-1]), 
                (f.shape[0], *tuple(cu.ones(f.ndim, dtype = cu.int8).tolist())))
        - (f[..., cu.newaxis, :].transpose(*tuple(cu.arange(0, f.ndim - 1, 1, dtype=cu.int8).tolist()), -1, -2) @ f[..., cu.newaxis, :])
        grad = gradient[..., cu.newaxis, :] @ soft
        return grad.reshape(self.inputs.shape) / self.x.shape[0]
    
class Sigmoid:
    def __call__(self, inputs, param=1.0) -> cu.ndarray:
        self.inputs = inputs
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            outputs = cu.empty(inputs.shape)
            sigmoid((inputs.shape[0],), (inputs.shape[1],), (inputs, param, outputs))
            return outputs
        return 1/(1+cu.power(cu.power(cu.exp(-inputs), param)))
    def backward(self, gradient) -> cu.ndarray:
        f = self(self.inputs)
        #if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
        #    outputs = cu.empty(f.shape)
        #    sigmoid_backward((f.shape[0],), (f.shape[1],), (f, outputs))
        #    return outputs 
        return gradient * (f * (1.0 - f))