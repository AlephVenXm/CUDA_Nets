import cupy as cu, math
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
        raise ValueError("One of values should be array. It is MATRIX`Add, not just add")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size != shape else x
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
        raise ValueError("One of values should be array. It is MATRIX`Subtract, not just subtract")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size != shape else x
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
        raise ValueError("One of values should be array. It is MATRIX`Multiply, not just multiply")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size != shape else x
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
        raise ValueError("One of values should be array. It is MATRIX`Divide, not just divide")
    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
    if dtype is None:
        dtype = x.dtype
    shape = max(x.shape, y.shape)
    rank = len(shape)
    z = cu.zeros(shape, dtype=dtype)
    PAD = lambda x : cu.full(shape, x) if x.size != shape else x
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def matdiv(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] / y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    matdiv[blocks, threads](PAD(x), PAD(y), z)
    return z
