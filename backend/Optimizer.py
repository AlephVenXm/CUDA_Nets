import os
if os.environ["USE_CUDA"] == "1":
    import cupy as cu
else:
    import numpy as cu
if os.environ["USE_NJIT"] == "1":
    from numba import njit
from Layer import Initializer

class Adam:
    def __init__(self,
                learning_rate: float = 0.001,
                beta1: float = 0.9,
                beta2: float = 0.99,
                epsilon: float = 10e-7,
                decay=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.bias_initializer = Initializer.Zeros()
    if os.environ["USE_NJIT"] == "1":
        @njit
        def update_weights(self, gradient, weights, v, m, v_hat, m_hat, t):
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * cu.power(gradient, 2)

            m_hat = m / (1 - cu.power(self.beta1, t))
            v_hat = v / (1 - cu.power(self.beta2, t))

            weights -= self.learning_rate * m_hat / (cu.sqrt(v_hat) + self.epsilon)

            return weights, v, m, v_hat, m_hat
    else:
        def update_weights(self, gradient, weights, v, m, v_hat, m_hat, t):
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * cu.power(gradient, 2)

            m_hat = m / (1 - cu.power(self.beta1, t))
            v_hat = v / (1 - cu.power(self.beta2, t))

            weights -= self.learning_rate * m_hat / (cu.sqrt(v_hat) + self.epsilon)

            return weights, v, m, v_hat, m_hat