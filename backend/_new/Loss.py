from Function import Function
import cupy as cu, math

class Loss:
    '''
    Base of Loss class
    '''
    def __init__(self,
                name: str="Loss",
                **kwargs):
        self.name = name
    def __call__(self, *args, **kwargs):
        '''
        Compute loss
        '''
        raise NotImplementedError(f"Method __call__() is not implemented for {self.name} class")
    def gradient(self, *args, **kwargs):
        '''
        Compute gradients of loss
        '''
        raise NotImplementedError(f"Method gradient() is not implemented for {self.name} class")
    
class MeanSquaredError(Loss):
    '''
    MeanSquaredError class

    Returns mean of square of subtract of targets and predicts

    MeanSquaredError(target, predict) = mean(pow(target - predict, 2))
    '''
    def __init__(self, 
                name: str="MeanSquaredError", 
                **kwargs):
        super().__init__(name=name, **kwargs)
        self.fn = Function("lambda t, p : math.pow(t - p, 2)")
        self.grad = Function("lambda t, p : -2 * (t - p)")
    def __call__(self, 
                target, 
                predict, 
                axis: int=-1, 
                keepdims: bool=True, 
                thread: int=10, 
                dtype=None):
        return cu.mean(self.fn(target, predict, thread=thread, dtype=dtype), axis=axis, keepdims=keepdims)
    def gradient(self, 
                target, 
                predict, 
                thread: int=10, 
                dtype=None):
        return cu.divide(self.grad(target, predict, thread=thread, dtype=dtype), cu.prod(target.shape[1:]))
    
class MeanAbsoluteError(Loss):
    '''
    MeanAbsoluteError class

    Returns mean of absolute of subtract of targets and predicts

    MeanAbsoluteError(target, predict) = mean(abs(target - predict, 2))
    '''
    def __init__(self, 
                name: str="MeanAbsoluteError", 
                **kwargs):
        super().__init__(name=name, **kwargs)
        self.fn = Function("lambda t, p : abs(t - p)")
        self.grad = Function("lambda t, p : -abs(t - p)")
    def __call__(self, 
                target, 
                predict, 
                axis: int=-1, 
                keepdims: bool=True, 
                thread: int=10, 
                dtype=None):
        return cu.mean(self.fn(target, predict, thread=thread, dtype=dtype), axis=axis, keepdims=keepdims)
    def gradient(self, 
                target, 
                predict, 
                thread: int=10, 
                dtype=None):
        return cu.divide(self.grad(target, predict, thread=thread, dtype=dtype), cu.prod(target.shape[1:]))

class CategoricalCrossentropy(Loss):
    '''
    CategoricalCrossentropy class

    Returns categorical loss of target and predict

    Uses with softmax activation

    CategoricalCrossentropy(target, predict) = -sum(target * log(predict))
    '''
    def __init__(self, 
                name: str="CategoricalCrossentropy", 
                **kwargs):
        super().__init__(name=name, **kwargs)
        self.fn = Function("lambda t, p : t * math.log(p)")
        self.grad = Function("lambda t, p : -t / (p)")
    def __call__(self, 
                target, 
                predict, 
                axis: int=-1, 
                keepdims: bool=True, 
                thread: int=10, 
                dtype=None):
        return -cu.sum(self.fn(target, predict, thread=thread, dtype=dtype), axis=axis, keepdims=keepdims)
    def gradient(self, 
                target, 
                predict, 
                thread: int=10, 
                dtype=None):
        return self.grad(target, predict, thread=thread, dtype=dtype)