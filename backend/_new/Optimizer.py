from Function import Function

class Optimizer:
    def __init__(self,
                learning_rate: float=0.001,
                name: str="Optimizer",
                **kwargs):
        self.learning_rate = learning_rate
        self.name = name
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Method __call__() is not implemented for {self.name} class")

class Adam(Optimizer):
    def __init__(self,
                learning_rate: float=0.001,
                beta1: float=0.9,
                beta2: float=0.99,
                epsilon: float=10e-7,
                name: str="Adam",
                **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_v_func = Function("lambda beta1, beta2, m, v, gradient : (beta1 * m + (1 - beta1) * gradient, beta2 * v + (1 - beta2) * math.pow(gradient, 2))")
        self.hat_func = Function("lambda m, v, beta1, beta2, t : (m / (1 - math.pow(beta1, t)), v / (1 - math.pow(beta2, t)))")
        self.w_func = Function("lambda w, lr, m_hat, v_hat, epsilon : w - lr * m_hat / (math.sqrt(v_hat) + epsilon)")
    def update_weights(self, gradient, weights, v, m, v_hat, m_hat, t, thread: int=10, dtype=None):
        m, v = self.m_v_func(self.beta1, self.beta2, m, v, gradient, thread=thread, dtype=dtype)
        m_hat, v_hat = self.hat_func(m, v, self.beta1, self.beta2, t, thread=thread, dtype=dtype)
        weights = self.w_func(weights, self.learning_rate, m_hat, v_hat, self.epsilon, thread=thread, dtype=dtype)
        return weights, v, m, v_hat, m_hat