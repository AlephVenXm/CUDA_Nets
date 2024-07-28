import os, math
if os.environ["USE_CUDA"] == "1":
    import cupy as cu
else:
    import numpy as cu
from numba import cuda

### ///////////////////////////////////////////////// ###
### /// OPERATIONS SUPPORT MATRIXES WITH MATRIXES /// ###
### ///      AND SINGLE VALUES WITH MATRIXES      /// ###
### ///////////////////////////////////////////////// ###

def MatAdd(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    '''
    Element Wisely adds first value to second

    X, Y -> Matrix or a single Value with rank < 4

    Thread -> number of Threads per one Block. Scales with max rank of Values

    Dtype -> dtype of Values in Return
    '''
    if x.size == 1 and y.size == 1:
        raise ValueError(f"One of values should be array. It is MATRIX`Add, not just add")
    if x.shape != y.shape and (x.size > 1 and y.size > 1):
        raise ValueError(f"Cannot add arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size == 1 else x
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def matadd(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] + y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    matadd[blocks, threads](PAD(x), PAD(y), z)
    return z

def MatSub(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    '''
    Element Wisely subtracts first value to second

    X, Y -> Matrix or a single Value with rank < 4

    Thread -> number of Threads per one Block. Scales with max rank of Values

    Dtype -> dtype of Values in Return
    '''
    if x.size == 1 and y.size == 1:
        raise ValueError(f"One of values should be array. It is MATRIX`Subtract, not just subtract")
    if x.shape != y.shape and (x.size > 1 and y.size > 1):
        raise ValueError(f"Cannot subtract arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size == 1 else x
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def matsub(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] - y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    matsub[blocks, threads](PAD(x), PAD(y), z)
    return z

def MatMul(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    '''
    Element Wisely multiplies first value to second

    X, Y -> Matrix or a single Value with rank < 4

    Thread -> number of Threads per one Block. Scales with max rank of Values

    Dtype -> dtype of Values in Return
    '''
    if x.size == 1 and y.size == 1:
        raise ValueError(f"One of values should be array. It is MATRIX`Multiply, not just multiply")
    if x.shape != y.shape and (x.size > 1 and y.size > 1):
        raise ValueError(f"Cannot multiply arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size == 1 else x
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def matmul(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] * y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    matmul[blocks, threads](PAD(x), PAD(y), z)
    return z

def MatDiv(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    '''
    Element Wisely divides first value to second

    X, Y -> Matrix or a single Value with rank < 4

    Thread -> number of Threads per one Block. Scales with max rank of Values

    Dtype -> dtype of Values in Return
    '''
    if x.size == 1 and y.size == 1:
        raise ValueError(f"One of values should be array. It is MATRIX`Divide, not just divide")
    if x.shape != y.shape and (x.size > 1 and y.size > 1):
        raise ValueError(f"Cannot divide arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size == 1 else x
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def matdiv(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] / y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    matdiv[blocks, threads](PAD(x), PAD(y), z)
    return z

def MatLinear(k, x, b, thread: int=10, dtype=None) -> cu.ndarray:
    '''
    Element Wise Linear function: kx + b

    K, X, B -> Matrix or a single Value with rank < 4

    Thread -> number of Threads per one Block. Scales with max rank of Values
    
    Dtype -> dtype of Values in Return
    '''
    if k.size == 1 and x.size == 1 and b.size == 1:
        raise ValueError(f"One of values should be array. It is MATRIX`Linear, not just linear")
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
