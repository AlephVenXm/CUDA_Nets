import cupy as cu, math
from numba import cuda

### //////////////////////////////////////////// ###
### /// STRUCT CUDA FUNCTION FROM ITS LAMBDA /// ###
### ///            LESS UNSTABLE             /// ###
### //////////////////////////////////////////// ###

class Function:
    '''
    Makes a Function class from python lambda 

    Function works on call of class

    Lambda function should be a string, e.g. "lambda x, y : x * y"

    Lambda can contain if/else statements, math module and any amount of values

    Lambda should not contain indexes, sums, prods and etc.

    Example ReLU:

    relu = Function("lambda x : x if x >= 0 else 0")
        or Function("lambda x : max(0, x)")
    
    x = cu.linspace(-1.0, 1.0, 10)

    y = relu(x, dtype=cu.float64)

    Example Linear:

    x = cu.linspace(-1.0, 1.0, 10)

    y = Function("lambda k, x, b : k*x+b")(x)
    '''
    def __init__(self, function):
        self.function = function
    def __call__(self, *args, thread: int=10, dtype=None):
        alph = ["val_" + f"{i}" for i in range(len(args))]
        values = ''.join([str(i) + ", " for i in alph[:len(args)]])[:-2]
        idx_values = ''.join([str(i) + "[idx], " for i in alph[:len(args)]])[:-2]
        padded_values = ''.join(["PAD(" + str(i) + "), " for i in alph[:len(args)]])[:-2]
        exec(f'''def func({values}, thread: int=10, dtype=None):
            if dtype is None:
                dtype = val_0.dtype
            shape = max({''.join([str(i) + ".shape, " for i in alph[:len(args)]])[:-2]}, ())
            rank = len(shape)
            result = cu.zeros(shape, dtype=dtype)
            PAD = lambda x : cu.full(shape, x) if x.shape != shape else x
            @cuda.jit(f'void({''.join(["{" + str(i) + ".dtype}[{':,'*rank}], " for i in alph[:len(args)]])[:-2]}, {"{result.dtype}"}[{"{':,'*rank}"}])')
            def function({values}, result):
               idx = cuda.grid(rank) 
               cfunc = {self.function}
               result[idx] = cfunc({idx_values})
            threads = (thread,) * rank
            blocks = tuple([math.ceil(result.shape[i] / threads[i]) for i in range(rank)])
            function[blocks, threads]({padded_values}, result)
            return result''')
        return eval(f'''func({''.join([f"args[{i}], " for i in range(len(args))])[:-2]}, thread=thread, dtype=dtype)''')

class AdvancedFunction:
    '''
    ... Dont ask me why

    This one ADVANCED`function able to return tuple of results

    e.g. AdvancedFunction("lambda x, y : (x + y, x - y)")(x, y)

    will return tuple of (res_0, res_1)

    Amount of inputs and output values can be any
    '''
    def __init__(self, function):
        self.function = function
        self.f = eval(function)
    def __call__(self, *args, thread: int=10, dtype=None):
        outputs = 1
        try: outputs = len(eval(f"self.f({''.join(["1, " for _ in range(len(args))])[:-2]})"))
        except: TypeError
        finally: ...
        val = ["val_" + f"{i}" for i in range(len(args))]
        res = ["res_" + f"{i}" for i in range(outputs)]
        values = ''.join([str(i) + ", " for i in val[:len(args)]])[:-2]
        results = ''.join([str(i) + ", " for i in res[:outputs]])[:-2]
        idx_values = ''.join([str(i) + "[idx], " for i in val[:len(args)]])[:-2]
        idx_results = ''.join([str(i) + "[idx], " for i in res[:outputs]])[:-2]
        padded_values = ''.join(["PAD(" + str(i) + "), " for i in val[:len(args)]])[:-2]
        exec(f'''def func({values}, thread: int=10, dtype=None):
            if dtype is None:
                dtype = val_0.dtype
            shape = max({''.join([str(i) + ".shape, " for i in val[:len(args)]])[:-2]}, ())
            rank = len(shape)
            {results} = {''.join(["cu.zeros(shape, dtype=dtype), " for _ in range(outputs)])[:-2]}
            PAD = lambda x : cu.full(shape, x) if x.shape != shape else x
            @cuda.jit(f'void({''.join(["{" + str(i) + ".dtype}[{':,'*rank}], " for i in val[:len(args)]])[:-2]}, {''.join(["{" + str(i) + ".dtype}[{':,'*rank}], " for i in res[:outputs]])[:-2]})')
            def function({values}, {results}):
               idx = cuda.grid(rank) 
               cfunc = {self.function}
               {idx_results} = cfunc({idx_values})
            threads = (thread,) * rank
            blocks = tuple([math.ceil(res_0.shape[i] / threads[i]) for i in range(rank)])
            function[blocks, threads]({padded_values}, {results})
            return {results}''')
        return eval(f'''func({''.join([f"args[{i}], " for i in range(len(args))])[:-2]}, thread=thread, dtype=dtype)''')

### //////////////////////////////////////// ###
### ///       ACTIVATION FUNCTIONS       /// ###
### /// https://arxiv.org/pdf/2109.14545 /// ###
### //////////////////////////////////////// ###

def Linear(k, x, b, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda k, x, b : k * x + b")(k, x, b, thread=thread, dtype=dtype)

def Tanh(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))")(x, thread=thread, dtype=dtype)

def sTanh(x, a, b, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, a, b : a * ((math.exp(b * x) - math.exp(b * -x)) / (math.exp(b * x) + math.exp(b * -x)))")(x, a, b, thread=thread, dtype=dtype)

def PSF(x, m, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, m : 1 / math.pow(1 + math.exp(-x), m)")(x, m, thread=thread, dtype=dtype)

def Sech(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : 2 / (math.exp(x) + math.exp(-x))")(x, thread=thread, dtype=dtype)

def ReSech(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : x * (2 / (math.exp(x) + math.exp(-x)))")(x, thread=thread, dtype=dtype)

def Sigmoid(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : 1 / (1 + math.exp(-x))")(x, thread=thread, dtype=dtype)

def sSigmoid(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : 4 * 1 / (1+math.exp(-x)) - 2")(x, thread=thread, dtype=dtype)

def pTanh(x, a, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, a : (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)) if x >= 0 else a * ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))")(x, a, thread=thread, dtype=dtype)

def Hexpo(x, a, b, c, d, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, a, b, c, d : -a * (math.exp(-x / b) - 1) if x >= 0 else c * (math.exp(x / d) - 1)")(x, a, b, c, d, thread=thread, dtype=dtype)

def SiLU(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : x * (1 / (1 + math.exp(-x)))")(x, thread=thread, dtype=dtype)

def ISigmoid(x, a, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, a : a * (x - a) + 1 / (1 + math.exp(-a)) if x >= a else (1 / (1 + math.exp(-x)) if -a < x < a else a * (x + a) + 1 / (1 + math.exp(-a)))")(x, a, thread=thread, dtype=dtype)

def LiSHT(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : x * ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))")(x, thread=thread, dtype=dtype)

def Elliott(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : (0.5 + x) / (1 + abs(x)) + 0.5")(x, thread=thread, dtype=dtype)

def SRS(x, a, b, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, a, b : x / (x / a + math.exp(-x / b))")(x, a, b, thread=thread, dtype=dtype)

def ReLU(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : x if x >= 0 else 0")(x, thread=thread, dtype=dtype)

def LReLU(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : x if x >= 0 else 0.01 * x")(x, thread=thread, dtype=dtype)

def PReLU(x, p, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, p : x if x >= 0 else p * x")(x, p, thread=thread, dtype=dtype)

def RReLU(x, low: float=0.0, high: float=1.0, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, rng : x if x >= 0 else rng * x")(x, cu.random.uniform(low, high, x.shape), thread=thread, dtype=dtype)

def CReLU(x, thread: int=10, dtype=None) -> list[cu.ndarray, cu.ndarray]:
    return [Function("lambda x : x if x >= 0 else 0")(x, thread=thread, dtype=dtype),
            Function("lambda x : 0 if x >= 0 else x")(x, thread=thread, dtype=dtype)]

def PTELU(x, a, b, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, a, b : x if x >= 0 else a * ((math.exp(b * x) - math.exp(b * -x)) / (math.exp(b * x) + math.exp(b * -x)))")(x, a, b, thread=thread, dtype=dtype)

def FReLU(x, b, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, b : x + b if x >= 0 else 0 + b")(x, b, thread=thread, dtype=dtype)

def RTReLU(x, a, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, a : x + a if (x + a) > 0 else 0")(x, a, thread=thread, dtype=dtype)

def ABReLU(x, b, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, b : max(0, x - b)")(x, b, thread=thread, dtype=dtype)

def DualReLU(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x, y : max(0, x) - max(0, y)")(x, y, thread=thread, dtype=dtype)

def PairedReLU():
    ...

def vReLU():
    ...

def DisReLU():
    ...

def BLU():
    ...

def L_ReLU():
    ...

def MTLU():
    ...

def EReLU():
    ...

def NLReLU():
    ...

def PLU():
    ...

def BReLU():
    ...

def ELU():
    ...

def SELU():
    ...

def PELU():
    ...

def CELU():
    ...

def MPELU():
    ...

def PREU():
    ...

def FELU():
    ...

def EELU():
    ...

def PDELU():
    ...

def ELiSH():
    ...

def HardELiSH():
    ...

def APL():
    ...

def ESwish():
    ...

def AAF():
    ...

def SLAF():
    ...

def MeLU():
    ...

def SAF():
    ...

def BDAA():
    ...

def TAF():
    ...

def SLU():
    ...

def RSP():
    ...

def Mish():
    ...

def GELU(x, thread: int=10, dtype=None) -> cu.ndarray:
    return Function("lambda x : 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*math.pow(x, 3))))")(x, thread=thread, dtype=dtype)

def RePU():
    ...

def PAU():
    ...

def Softmax(x, axis: int=-1,thread: int=10, dtype=None) -> cu.ndarray:
    numerator = Function("lambda x, y : math.exp(x - y)")(x, cu.max(x, axis=axis, keepdims=True), thread=thread, dtype=dtype)
    return Function("lambda x, y : x / y")(numerator, cu.sum(numerator, axis=axis, keepdims=True), thread=thread, dtype=dtype)
