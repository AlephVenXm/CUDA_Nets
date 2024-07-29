import cupy as cu, math
from numba import cuda

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
    if (k.shape != x.shape or k.shape != b.shape or b.shape != x.shape) and (k.size > 1 and x.size > 1 and b.size > 1):
        raise ValueError(f"Cannot operate with arrays of different shapes: Got arrays of shape: {k.shape}, {x.shape}, {b.shape}")
    if len(k.shape) > 3 or len(x.shape) > 3 or len(b.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(k.shape)}, {len(x.shape)}, {len(b.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(k.shape, x.shape, b.shape)
    rank = len(shape)
    y = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size == 1 else x
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
        y[idx] = (math.exp(x[idx]) - math.exp(-x[idx])) / (math.exp(x[idx] + math.exp(x[idx])))
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
        y[idx] = a * ((math.exp(b * x[idx]) - math.exp(b * -x[idx])) / (math.exp(b * x[idx] + math.exp(b * x[idx]))))
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
            y[idx] = (math.exp(x[idx]) - math.exp(-x[idx])) / (math.exp(x[idx] + math.exp(x[idx])))
        else:
            y[idx] = a * ((math.exp(x[idx]) - math.exp(-x[idx])) / (math.exp(x[idx] + math.exp(x[idx]))))
    threads = (thread,) * rank
    blocks = tuple([math.ceil(y.shape[i] / threads[i]) for i in range(rank)])
    ptanh[blocks, threads](x, a, y)
    return y

def Hexpo():
    ...

def SiLU():
    ...

def ISigmoid():
    ...

def LiSHT():
    ...

def Elliott():
    ...

def SRS():
    ...

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

def LReLU():
    ...

def PReLU():
    ...

def RReLU():
    ...

def CReLU():
    ...

def FReLU():
    ...

def RTReLU():
    ...

def ABReLU():
    ...

def DualReLU():
    ...

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
