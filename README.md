# CUDA_Nets
[WIP] This repo implements various AI and ML functions with CUDA architecture on Python, uniting simplicity of Python and parallel computations with CUDA

# Uniting Python and CUDA

![merge](https://github.com/AlephVenXm/CUDA_Nets/blob/main/merge.png)

# Test on function = sum(sqrt(linspace(0.0, 100.0, 10e6)))

![compare](https://github.com/AlephVenXm/CUDA_Nets/blob/main/compare.png)

# Random-walk test

>> Cuda

433 μs ± 71.2 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)

>> Numpy

9.4 ms ± 208 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>> TensorFlow

17.6 ms ± 53.2 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
