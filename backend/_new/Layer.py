import cupy as cu

class Layer:
    '''
    Base of all layers
    '''
    def __init__(self,
                shape=None,
                trainable: bool=True,
                dtype=None,
                name: str="Layer",
                **kwargs):
        self.shape = shape
        self.trainable = trainable
        self.dtype = dtype
        self.name = name
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Method __call__() is not implemented for {self.name} class")
    def add_weights(self,
                    input_shape=None,
                    output_shape=None,
                    kernel_initializer=None,
                    bias_initializer=None,
                    use_bias: bool=True,
                    dtype=None):
        if kernel_initializer is not None:
            self.weights = kernel_initializer(input_shape[-1], output_shape, dtype=dtype)
        else:
            self.weights = cu.random.uniform(0.0, 1.0, size=(input_shape[-1], output_shape), dtype=dtype)
        if use_bias:
            if bias_initializer is not None:
                self.biases = bias_initializer(output_shape, dtype=dtype)
            else:
                self.biases = cu.zeros(output_shape, dtype=dtype)
        self.shape = self.weights.shape
    def get_weights(self):
        return self.weights
    def get_biases(self):
        return self.biases
    def set_weights(self, weights):
        self.weights = weights
    def set_biases(self, biases):
        self.biases = biases
    def debug(self):
        ...

class Input(Layer):
    '''
    Input layer

    Used with other layers to set their input shape

    Shape must be tuple e.g. (1,), (16, 16, 3)
    '''
    def __init__(self, 
                shape=None, 
                dtype=None, 
                name: str="Input", 
                **kwargs):
        super().__init__(shape=shape, dtype=dtype, name=name, **kwargs)
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Method __call__() is not implemented for {self.name} class")

class Dense(Layer):
    '''
    Densely connected neurons

    Should be built with build() method
    '''
    def __init__(self,
                units: int,
                shape=None,
                activation=None,
                kernel_initializer=None,
                bias_initializer=None,
                use_bias: bool=True,
                dtype=None,
                name: str="Dense",
                **kwargs):
        super().__init__(shape=shape, dtype=dtype, name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer= bias_initializer
        self.use_bias = use_bias
        self.built = False
    def build(self, inputs):
        if self.dtype is None:
            self.dtype = inputs.dtype
        if not self.built:
                self.add_weights(inputs.shape,
                                self.units,
                                self.kernel_initializer,
                                self.bias_initializer,
                                self.use_bias,
                                self.dtype)
        return self
    def __call__(self, inputs):
        self.in_data = inputs
        outputs = cu.add(cu.matmul(inputs, self.weights), self.biases)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    def backward(self, loss):
        gradient = cu.dot(cu.transpose(self.weights), loss)
        self.gradient_weights = cu.transpose(cu.dot(cu.transpose(loss), self.in_data))
        self.gradient_biases = cu.sum(loss, axis = 0)
        if self.activation is not None:
            return self.activation.backward(gradient)
        return gradient
    def get_weight_gradient(self):
        return self.gradient_weights
    def get_bias_gradient(self):
        return self.gradient_biases
    def set_weight_gradient(self, gradient):
        self.gradient_weights = gradient
    def set_bias_gradient(self, gradient):
        self.gradient_biases = gradient