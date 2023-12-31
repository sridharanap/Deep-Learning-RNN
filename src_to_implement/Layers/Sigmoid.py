import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = 1/(1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        activations_derivative = self.activations * (1 - self.activations)
        error_tensor = error_tensor * activations_derivative
        return error_tensor
