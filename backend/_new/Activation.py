from Function import ReLU as relu, GELU as gelu, Softmax as softmax, Function
import cupy as cu

class Activation:
    def __init__(self, name: str="Activation", dtype=None, **kwargs):
        self.name = name
        self.dtype = dtype
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Method __call__() is not implemented for {self.name} class")

class ReLU(Activation):
    def __init__(self, name: str="ReLU", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.fn = Function("lambda x : 1 if x >= 0 else 0")
    def __call__(self, x, thread: int=10):
        self.in_data = x
        return relu(x, thread=thread, dtype=self.dtype)
    def backward(self, loss):
        return cu.multiply(self.fn(self.in_data), loss)

class GELU(Activation):
    def __init__(self, name: str="GELU", **kwargs):
        super().__init__(name=name, **kwargs)
    def __call__(self, x, thread: int=10, dtype=None):
        return gelu(x, thread=thread, dtype=dtype)
    
class Softmax(Activation):
    def __init__(self, name: str="Softmax", **kwargs):
        super().__init__(name=name, **kwargs)
    def __call__(self, x, thread: int=10, dtype=None):
        return softmax(x, thread=thread, dtype=dtype)