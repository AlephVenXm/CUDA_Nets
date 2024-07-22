# CUDA_Nets [WORK IN PROGRESS]
[WIP] This repo implements various AI and ML functions with CUDA architecture on Python, uniting simplicity of Python and parallel computations with CUDA

Main idea: keras-style API with some pytorch additions, and, possibly, some realizations of self-adaptive algorithms (e.g. *fixed* DynamicGradient, LogicMemoryUnit, AdaptiveNeuralConnections, Multi-Dimensional Weight Access-Storage System...)

Current contributor(s): Aleph

# Uniting Python and CUDA

Interface of classes are written on Python, while all computation goes trough GPU using CUDA architecture

![merge](https://github.com/AlephVenXm/CUDA_Nets/blob/main/merge.png)

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
import cupy as cu
from Implementations import *
#Inputs
input_k = cu.random.randint(0, 10, 2).reshape(2, 1)
input_v = cu.random.randint(0, 10, 2).reshape(2, 1)
input_k_v = cu.append(input_k, input_v).reshape(2, 2)
input_q = cu.random.randint(0, 10, 2).reshape(2, 1)
#Attention block
attn = MultiHeadAttention(1, 1, 1, 1)(input_q, input_k, input_v)
attn_norm = Normalization()(attn)
attn_block = Add()([attn, attn_norm])
#Feed-Forward Net block
ffn = Dense(2, 10, activation=GeLU())(input_k_v)
ffn = Dense(10, 10, activation=GeLU())(ffn)
ffn_norm = Normalization()(ffn)
ffn_block = Add()([ffn, ffn_norm])
#Output probabilities
out = Add()([ffn_block, attn_block])
out = Dense(10, 10)(out)
out = Dense(10, 2, activation=Softmax())(out)

[[[ 0.49919651  0.50080349]
  [ 0.49929742  0.50070258]]

 [[ 0.00796389 -0.01614334]
  [ 0.00102635  0.01026686]]]
```
