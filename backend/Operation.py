import os, math
if os.environ["USE_CUDA"] == "1":
    import cupy as cu
else:
    import numpy as cu
from numba import cuda

### //////////////////////////////////////// ###
### /// OPERATIONS SUPPORT MATRIXES ONLY /// ###
### //////////////////////////////////////// ###
def Add(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    if x.shape != y.shape:
        print(f"Cannot add arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
        raise ValueError
    if len(x.shape) > 3 or len(y.shape) > 3:
        print(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
        raise ValueError
    if dtype is None:
        dtype=x.dtype
    z = cu.zeros(x.shape, dtype=dtype)
    rank = len(x.shape)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def add(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] + y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    add[blocks, threads](x, y, z)
    return z

def Sub(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    if x.shape != y.shape:
        print(f"Cannot subtract arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
        raise ValueError
    if len(x.shape) > 3 or len(y.shape) > 3:
        print(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
        raise ValueError
    if dtype is None:
        dtype=x.dtype
    z = cu.zeros(x.shape, dtype=dtype)
    rank = len(x.shape)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def sub(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] - y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    sub[blocks, threads](x, y, z)
    return z

def Mul(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    if x.shape != y.shape:
        print(f"Cannot multiply arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
        raise ValueError
    if len(x.shape) > 3 or len(y.shape) > 3:
        print(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
        raise ValueError
    if dtype is None:
        dtype=x.dtype
    z = cu.zeros(x.shape, dtype=dtype)
    rank = len(x.shape)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def mul(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] * y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    mul[blocks, threads](x, y, z)
    return z

def Div(x, y, thread: int=10, dtype=None) -> cu.ndarray:
    if x.shape != y.shape:
        print(f"Cannot divide arrays of different shapes: Got arrays of shape: {x.shape}, {y.shape}")
        raise ValueError
    if len(x.shape) > 3 or len(y.shape) > 3:
        print(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(x.shape)}, {len(y.shape)}")
        raise ValueError
    if dtype is None:
        dtype=x.dtype
    z = cu.zeros(x.shape, dtype=dtype)
    rank = len(x.shape)
    @cuda.jit(f'void({x.dtype}[{':,'*rank}], {y.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def div(x, y, z):
        idx = cuda.grid(rank)
        z[idx] = x[idx] / y[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    div[blocks, threads](x, y, z)
    return z

def Linear(k, x, b, thread: int=10, dtype=None) -> cu.ndarray:
    if k.shape != x.shape or k.shape != b.shape or b.shape != x.shape:
        print(f"Cannot operate with arrays of different shapes: Got arrays of shape: {k.shape}, {x.shape}, {b.shape}")
        raise ValueError
    if len(k.shape) > 3 or len(x.shape) > 3 or len(b.shape) > 3:
        print(f"Cannot operate with arrays with rank >= 4: Got arrays of rank: {len(k.shape)}, {len(x.shape)}, {len(b.shape)}")
        raise ValueError
    if dtype is None:
        dtype=x.dtype
    z = cu.zeros(x.shape, dtype=dtype)
    rank = len(x.shape)
    @cuda.jit(f'void({k.dtype}[{':,'*rank}], {x.dtype}[{':,'*rank}], {b.dtype}[{':,'*rank}], {z.dtype}[{':,'*rank}])')
    def linear(k, x, b, z):
        idx = cuda.grid(rank)
        z[idx] = k[idx] * x[idx] + b[idx]
    threads = (thread,) * rank
    blocks = tuple([math.ceil(z.shape[i] / threads[i]) for i in range(rank)])
    linear[blocks, threads](k, x, b, z)
    return z
