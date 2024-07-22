import cupy as cu

#Basic operations
div = cu.ElementwiseKernel(
    'float64 x, float64 y',
    'float64 z',
    'z = x / y',
    'div'
)
mul = cu.ElementwiseKernel(
    'float64 x, float64 y',
    'float64 z',
    'z = x * y',
    'mul'
)
add = cu.ElementwiseKernel(
    'float64 x, float64 y',
    'float64 z',
    'z = x + y',
    'add'
)
sub = cu.ElementwiseKernel(
    'float64 x, float64 y',
    'float64 z',
    'z = x - y',
    'sub'
)

#Activations
relu = cu.RawKernel(r'''
extern "C" __global__
void relu(const double* x, double* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (x[i] < 0) y[i] = 0;
    else y[i] = x[i];
}
''', 'relu')
gelu = cu.RawKernel(r'''
const double PI = 3.14159265359;
extern "C" __global__
void gelu(const double* x, double* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    y[i] = 0.5*x[i]*(1+tanh(sqrt(2/PI)*(x[i]+0.044715*pow(x[i], 3))));
}
''', 'gelu')
softmax = cu.RawKernel(r'''                
const double EXP = 2.71828182845;
extern "C" __global__
void softmax(const double* x, double* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double numerator[100]; //! This one over here should be pre-configurated
    double denominator = 0.0;
    double max = x[blockDim.x*((int)i/blockDim.x)];
    // Find max
    for (size_t c = 0; c <= blockDim.x; ++c)
        if (x[c+blockDim.x*((int)i/blockDim.x)] > max) max = x[c+blockDim.x*((int)i/blockDim.x)];
    // Find numerator
    for (size_t c = 0; c < blockDim.x; ++c) {
        numerator[c] = pow(EXP, x[c+blockDim.x*((int)i/blockDim.x)] - max);
        denominator += numerator[c];
    }
    for (size_t c = 0; c <= blockDim.x; ++c) 
        numerator[c] /= denominator;
    y[i] = numerator[threadIdx.x];
}
''', 'softmax')
sigmoid = cu.RawKernel(r'''
const double EXP = 2.71828182845;
extern "C" __global__
void sigmoid(const double* x, double param, double* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    y[i] = 1/(1+pow(pow(EXP, -x[i]), param));
}
''', 'sigmoid')

#Functional layers
normalization = cu.RawKernel(r'''
extern "C" __global__
void normalization(const double* in, const double* avg, const double* var, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    out[i] = (in[i]-avg[0])/(sqrt(var[0]));
}
''', 'normalization')
