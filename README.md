# CUDA_Nets [WORK IN PROGRESS]
[WIP] This repo implements various AI and ML functions with CUDA architecture on Python, uniting simplicity of Python and parallel computations with CUDA

Main idea: keras-style API with some pytorch additions, and, possibly, some realizations of self-adaptive algorithms (e.g. *fixed* DynamicGradient, LogicMemoryUnit, AdaptiveNeuralConnections, Multi-Dimensional Weight Access-Storage System...)

Current contributor(s): Aleph (**I'm currently studying Reinforcement Learning, this repo might freeze for a bit**)

# Uniting Python and CUDA

Interface of classes are written on Python, while all computation goes trough GPU using CUDA architecture

![merge](https://github.com/AlephVenXm/CUDA_Nets/blob/main/merge.png)

# Function class

```ruby
func = Function("lambda x, y : (x + y, x - y) if x > y else (x * y, x / y)")
x = cu.linspace(-1.0, 1.0, 4).reshape(2, 2)
y = cu.random.uniform(-1.0, 1.0, 2)
z1, z2 = func(x, y, thread=32, dtype=cu.float16)
print("first matrix:\n{}\nsecond matrix:\n{}".format(z1, z2))

first matrix:
[[-0.255  -0.2257]
 [ 0.588   1.677 ]]
second matrix:
[[-3.926   -0.4924 ]
 [ 0.07855  0.323  ]]
```

# Test on function = sum(sqrt(linspace(0.0, 100.0, 10e6)))

![compare](https://github.com/AlephVenXm/CUDA_Nets/blob/main/compare.png)

# Random-walk test

```ruby
def random_walk(n):
    steps = random.choice([-1,+1], n)
    return cumsum(steps)
%timeit walk = random_walk(10e5)
```

>> Cuda

433 μs ± 71.2 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)

>> Numpy

9.4 ms ± 208 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>> TensorFlow

17.6 ms ± 53.2 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Current progress

```ruby
import cupy as cu, matplotlib.pyplot as plt
from Implementations import *
#Input
q = cu.random.randint(0, 10, 10).reshape(10, 1)
k = cu.random.randint(0, 10, 10).reshape(10, 1)
v = cu.random.randint(0, 10, 10).reshape(10, 1)
#Embedding
e_q = Embedding(1, 1)(q)
e_k = Embedding(1, 1)(k)
e_v = Embedding(1, 1)(v)
#Attention
attn = MultiHeadAttention(1, 1)(e_q, e_k, e_v)
attn_norm = Normalization()(attn)
attn_block = Add()([attn, attn_norm])
#Feed-Forward
ffn = Dense(10, 10, activation=Activation.GeLU())(attn_block)
ffn = Dense(10, 10, activation=Activation.GeLU())(ffn)
ffn = Dense(10, 10, activation=Activation.GeLU())(ffn)
ffn_norm = Normalization()(ffn)
ffn_block = Add()([ffn, ffn_norm])
#Probabilities
lin = Dense(10, 10)(ffn_block)
soft = Dense(10, 10, activation=Activation.Softmax())(lin)

print(soft)

[[[1.51898902e-03 9.29503220e-04 1.36022515e-03 1.33556020e-03
   1.21412477e-03 1.03211580e-02 1.30989742e-03 3.34207198e-03
   9.78473213e-01 1.95257365e-04]
  [1.39390225e-03 1.08939315e-03 1.87080116e-03 3.10024725e-03
   9.85254421e-01 1.27299659e-03 1.19052687e-03 1.27049041e-03
   3.33600168e-03 2.21219257e-04]
    ...
```
