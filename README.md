# CUDA_Nets
[WIP] This repo implements various AI and ML functions with CUDA architecture on Python, uniting simplicity of Python and parallel computations with CUDA

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
