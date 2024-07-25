import os
if os.environ["USE_CUDA"] == "1":
    import cupy as cu
    if os.environ["USE_C"] == "1":
        from C_Functions import sub, div, mul
else:
    import numpy as cu
from Activation import Softmax

class MeanAbsoluteError:
    def __call__(self, inputs, targets):
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            return sub(targets, inputs)
        return targets - inputs
    def derivative(self, inputs, targets):
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            return -1 * (sub(targets, inputs)) / cu.prod(inputs.shape[1:])
        return -1 * cu.abs(targets - inputs) / cu.prod(inputs.shape[1:])

class MeanSquaredError:
    def __call__(self, inputs, targets):
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            return cu.power(sub(targets, inputs), 2)
        return cu.power(targets - inputs, 2)
    def derivative(self, inputs, targets):
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            return -2 * (sub(targets, inputs)) / cu.prod(inputs.shape[1:])
        return -2 * (targets - inputs) / cu.prod(inputs.shape[1:])

class CrossEntropy:
    def __init__(self) -> None:
        self.softmax = Softmax()
    def __call__(self, inputs, targets):
        softmax = cu.log(self.softmax(inputs))
        loss = -softmax[cu.arange(len(targets)), targets]
        return loss
    def derivative(self, inputs, targets):
        loss = 1/inputs.shape[0]
        loss_derivative = -1 * cu.where(cu.isin(inputs, inputs[cu.arange(len(targets)), targets]), loss)
        output_loss = self.softmax.backward(loss_derivative)
        return output_loss

class CategoricalCrossEntropy:
    def __call__(self, inputs, targets):
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            return mul(-targets, cu.log(inputs))
        return -targets * cu.log(inputs)
    def derivative(self, inputs, targets):
        if os.environ["USE_CUDA"] == "1" and os.environ["USE_C"] == "1":
            return div(-targets, inputs)
        return -targets / inputs
