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
        for i in range(len(args)):
            exec(f"{alph[i]} = args[i]")
        values = ''.join([str(i) + ", " for i in alph[:len(args)]])[:-2]
        idx_values = ''.join([str(i) + "[idx], " for i in alph[:len(args)]])[:-2]
        padded_values = ''.join(["PAD(" + str(i) + "), " for i in alph[:len(args)]])[:-2]
        exec(f'''def func({values}, thread: int=10, dtype=None):
            if dtype is None:
                dtype = val_1.dtype
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
        return eval(f'''func({values}, thread=thread, dtype=dtype)''')


### //////////////////////////////////////// ###
### ///       ACTIVATION FUNCTIONS       /// ###
### /// https://arxiv.org/pdf/2109.14545 /// ###
### //////////////////////////////////////// ###

def Linear(k, x, b, thread: int=10, dtype=None) -> cu.ndarray:
    '''
    Element Wise Linear function: kx + b

    K, X, B -> Matrix or a single Value with rank < 4

    Thread -> number of Threads per one Block. Scales with max rank of Values
    
    Dtype -> dtype of Values in Return
    '''
    if k.size == 1 and x.size == 1 and b.size == 1:
        raise ValueError("One of values should be array")
    if len(k.shape) > 3 or len(x.shape) > 3 or len(b.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(k.shape)}, {len(x.shape)}, {len(b.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(k.shape, x.shape, b.shape)
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.shape != shape else x
    @cuda.jit(f'void({k.dtype}[{':,'*rank}], {x.dtype}[{':,'*rank}], {b.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def linear(k, x, b, y):
        idx = cuda.grid(rank)
        y[idx] = k[idx] * x[idx] + b[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    linear[blocks, threads](PAD(k), PAD(x), PAD(b), y)
    return y

def Tanh(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def tanh(x, y):
        idx = cuda.grid(rank)
        y[idx] = (math.exp(x[idx]) - math.exp(-x[idx])) / (math.exp(x[idx]) + math.exp(-x[idx]))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    tanh[blocks, threads](x, y)
    return y

def sTanh(x, a, b, thread: int=10, dtype=None) -> cu.ndarray:
    if a.size > 1 or b.size > 1:
        raise ValueError("Parameters of sTanh should be single values, not arrays")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {a.dtype}, {b.dtype}, {y.dtype}[{':,'*rank}])')
    def stanh(x, a, b, y):
        idx = cuda.grid(rank)
        y[idx] = a * ((math.exp(b * x[idx]) - math.exp(b * -x[idx])) / (math.exp(b * x[idx]) + math.exp(b * -x[idx])))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    stanh[blocks, threads](x, a, b, y)
    return y

def PSF(x, m, thread: int=10, dtype=None) -> cu.ndarray:
    if m.size > 1:
        raise ValueError("Parameter of PSF should be single value, not array")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {m.dtype}, {y.dtype}[{':,'*rank}])')
    def PSF(x, m, y):
        idx = cuda.grid(rank)
        y[idx] = 1 / math.pow(1 + math.exp(-x[idx]), m)
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    PSF[blocks, threads](x, m, y)
    return y

def Sech(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def sech(x, y):
        idx = cuda.grid(rank)
        y[idx] = 2 / (math.exp(x[idx]) + math.exp(-x[idx]))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    sech[blocks, threads](x, y)
    return y

def ReSech(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def resech(x, y):
        idx = cuda.grid(rank)
        y[idx] = x[idx] * (2 / (math.exp(x[idx]) + math.exp(-x[idx])))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    resech[blocks, threads](x, y)
    return y

def sSigmoid(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def ssigmoid(x, y):
        idx = cuda.grid(rank)
        y[idx] = 4 * 1/(1+math.exp(-x[idx])) - 2
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    ssigmoid[blocks, threads](x, y)
    return y

def pTanh(x, a, thread: int=10, dtype=None) -> cu.ndarray:
    if a.size > 1:
        raise ValueError("Parameter of pTanh should be single value, not array")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {a.dtype}, {y.dtype}[{':,'*rank}])')
    def ptanh(x, a, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = (math.exp(x[idx]) - math.exp(-x[idx])) / (math.exp(x[idx]) + math.exp(-x[idx]))
        else:
            y[idx] = a * ((math.exp(x[idx]) - math.exp(-x[idx])) / (math.exp(x[idx]) + math.exp(-x[idx])))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    ptanh[blocks, threads](x, a, y)
    return y

def Hexpo(x, a, b, c, d, thread: int=10, dtype=None) -> cu.ndarray:
    if a.size > 1 or b.size > 1 or c.size > 1 or d.size > 1:
        raise ValueError("Parameters of Hexpo should be single values, not arrays")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {a.dtype}, {b.dtype}, {c.dtype}, {d.dtype}, {y.dtype}[{':,'*rank}])')
    def hexpo(x, a, b, c, d, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = -a * (math.exp(-x[idx]/b) - 1)
        else:
            y[idx] = c * (math.exp(x[idx]/d) - 1)
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    hexpo[blocks, threads](x, a, b, c, d, y)
    return y

def SiLU(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def silu(x, y):
        idx = cuda.grid(rank)
        y[idx] = x[idx] * 1/(1+math.exp(-x[idx]))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    silu[blocks, threads](x, y)
    return y

def ISigmoid(x, a, thread: int=10, dtype=None) -> cu.ndarray:
    if a.size > 1:
        raise ValueError("Parameter of ISigmoid should be single value, not array")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {a.dtype}, {y.dtype}[{':,'*rank}])')
    def isigmoid(x, a, y):
        idx = cuda.grid(rank)
        if x[idx] >= a:
            y[idx] = a * (x[idx] - a) + 1/(1+math.exp(-a))
        elif -a < x[idx] < a:
            y[idx] = 1/(1+math.exp(-x[idx]))
        else:
            y[idx] = a * (x[idx] + a) + 1/(1+math.exp(-a))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    isigmoid[blocks, threads](x, a, y)
    return y

def LiSHT(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def lisht(x, y):
        idx = cuda.grid(rank)
        y[idx] = x[idx] * ((math.exp(x[idx]) - math.exp(-x[idx])) / (math.exp(x[idx]) + math.exp(-x[idx])))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    lisht[blocks, threads](x, y)
    return y

def Elliott(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def elliott(x, y):
        idx = cuda.grid(rank)
        y[idx] = (0.5 + x[idx]) / (1 + abs(x[idx])) + 0.5
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    elliott[blocks, threads](x, y)
    return y

def SRS(x, a, b, thread: int=10, dtype=None) -> cu.ndarray:
    if a.size > 1 or b.size > 1:
        raise ValueError("Parameters of SRS should be single values, not arrays")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {a.dtype}, {b.dtype}, {y.dtype}[{':,'*rank}])')
    def srs(x, a, b, y):
        idx = cuda.grid(rank)
        y[idx] = x[idx] / (x[idx] / a + math.exp(-x[idx]/b))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    srs[blocks, threads](x, a, b, y)
    return y

def ReLU(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def relu(x, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = x[idx]
        else:
            y[idx] = 0
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    relu[blocks, threads](x, y)
    return y

def LReLU(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def lrelu(x, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = x[idx]
        else:
            y[idx] = 0.01 * x[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    lrelu[blocks, threads](x, y)
    return y

def PReLU(x, p, thread: int=10, dtype=None) -> cu.ndarray:
    if p.size > 1:
        raise ValueError("Parameters of PReLU should be single value, not array")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {p.dtype}, {y.dtype}[{':,'*rank}])')
    def prelu(x, p, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = x[idx]
        else:
            y[idx] = p * x[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    prelu[blocks, threads](x, p, y)
    return y

def RReLU(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def rrelu(x, rng, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = x[idx]
        else: 
            y[idx] = rng[idx] * x[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    rng = cu.random.uniform(0.0, 1.0, shape, dtype=dtype)
    rrelu[blocks, threads](x, rng, y)
    return y

def CReLU(x, thread: int=10, dtype=None) -> list[cu.ndarray, cu.ndarray]:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y_positive = cu.zeros(shape, dtype=dtype)
    y_negative = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y_positive.dtype}[{':,'*rank}], {y_negative.dtype}[{':,'*rank}])')
    def crelu(x, y_positive, y_negative):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y_positive[idx] = x[idx]
            y_negative[idx] = 0
        else:
            y_positive[idx] = 0
            y_negative[idx] = x[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(shape[i] / threads[i]) for i in range(rank)])
    crelu[blocks, threads](x, y_positive, y_negative)
    return [y_positive, y_negative]

def PTELU(x, a, b, thread: int=10, dtype=None) -> cu.ndarray:
    if a.size > 1 or b.size > 1:
        raise ValueError("Parameters of PTELU should be single values, not arrays")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {a.dtype}, {b.dtype}, {y.dtype}[{':,'*rank}])')
    def ptelu(x, a, b, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = x[idx]
        else:
            y[idx] = a * ((math.exp(b * x[idx]) - math.exp(b * -x[idx])) / (math.exp(b * x[idx]) + math.exp(b * -x[idx])))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    ptelu[blocks, threads](x, a, b, y)
    return y

def FReLU(x, b, thread: int=10, dtype=None) -> cu.ndarray:
    if b.size > 1:
        raise ValueError("Parameters of FReLU should be single value, not array")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {b.dtype}, {y.dtype}[{':,'*rank}])')
    def frelu(x, b, y):
        idx = cuda.grid(rank)
        if x[idx] >= 0:
            y[idx] = x[idx] + b
        else:
            y[idx] = 0 + b
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    frelu[blocks, threads](x, b, y)
    return y

def RTReLU(x, a, thread: int=10, dtype=None) -> cu.ndarray:
    if a.size > 1:
        raise ValueError("Parameters of RTReLU should be single value, not array")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {a.dtype}, {y.dtype}[{':,'*rank}])')
    def rtrelu(x, a, y):
        idx = cuda.grid(rank)
        if x[idx] + a > 0:
            y[idx] = x[idx] + a
        else:
            y[idx] = 0
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    rtrelu[blocks, threads](x, a, y)
    return y

def ABReLU(x, b, thread: int=10, dtype=None) -> cu.ndarray:
    if b.size > 1:
        raise ValueError("Parameters of ABReLU should be single value, not array")
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {b.dtype}, {y.dtype}[{':,'*rank}])')
    def abrelu(x, b, y):
        idx = cuda.grid(rank)
        y[idx] = max(0, x[idx] - b)
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    abrelu[blocks, threads](x, b, y)
    return y

def DualReLU(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def dualrelu(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = max(0, x[idx]) - max(0, y[idx])   
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    dualrelu[blocks, threads](x, y, z)
    return z

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
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def gelu(x, y):
        idx = cuda.grid(rank)
        y[idx] = 0.5*x[idx]*(1+math.tanh(math.sqrt(2/math.pi)*(x[idx]+0.044715*math.pow(x[idx], 3))))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    gelu[blocks, threads](x, y)
    return y

def RePU():
    ...

def PAU():
    ...

def Softmax(x, thread: int=10, dtype=None) -> cu.ndarray:
    if len(x.shape) > 3:
        raise ValueError(f"Cannot operate with array with rank >= 4: Got array of rank: {len(x.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = x.shape
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    numerator = cu.zeros(shape, dtype=dtype)
    maximum = cu.max(x, axis=-1, keepdims=True)
    PAD = lambda x : cu.full(shape, x) if x.shape != shape else x
    @cuda.jit(f'void({numerator.dtype}[{':,'*rank}], {maximum.dtype}[{':,'*rank}])')
    def find_numerator(numerator, maximum):
        idx = cuda.grid(rank)
        numerator[idx] = math.exp(numerator[idx] - maximum[idx])
    @cuda.jit(f'void({numerator.dtype}[{':,'*rank}], {numerator.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}])')
    def softmax(numerator, sum_numerator, y):
        idx = cuda.grid(rank)
        y[idx] = numerator[idx] / sum_numerator[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    find_numerator[blocks, threads](numerator, PAD(maximum))
    softmax[blocks, threads](numerator, PAD(cu.sum(numerator, axis=-1, keepdims=True)), y)
    return y
