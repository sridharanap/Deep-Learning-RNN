import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.updated_weight = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        if self.regularizer:
            self.updated_weight = self.weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)) - (self.learning_rate * self.gradient_tensor)

        else:
            self.updated_weight = self.weight_tensor - (self.learning_rate * self.gradient_tensor)
        return self.updated_weight


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor
        vk1 = (self.momentum_rate * self.vk) - (self.learning_rate * self.gradient_tensor)
        if self.regularizer:
            self.updated_weight = self.weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(self.weight_tensor))
            self.updated_weight = self.updated_weight + vk1
        else:
            self.updated_weight = self.weight_tensor + vk1
        self.vk = vk1
        return self.updated_weight


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.vk = 0
        self.rk = 0
        self.k = 1
        self.updated_weight = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.weight_tensor=weight_tensor
        self.vk = self.mu * self.vk + (1.0 - self.mu) * gradient_tensor
        self.rk = self.rho * self.rk + (1.0 - self.rho) * gradient_tensor * gradient_tensor
        self.mu_k = np.power(self.mu, self.k)
        self.rho_k = np.power(self.rho, self.k)
        vk_bias = self.vk / (1.0 - self.mu_k)
        rk_bias = self.rk / (1.0 - self.rho_k)
        self.k += 1
        if self.regularizer:
            self.updated_weight = self.weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
            self.updated_weight = self.updated_weight - self.learning_rate * vk_bias / (np.sqrt(rk_bias) + np.finfo(float).eps)
        else:
            self.updated_weight = weight_tensor - self.learning_rate * vk_bias / (np.sqrt(rk_bias) + np.finfo(float).eps)
        return self.updated_weight
