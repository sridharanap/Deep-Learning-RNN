import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        activations_derivative = 1 - np.square(self.activations)
        error_tensor = error_tensor * activations_derivative
        return error_tensor
