from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.ip1 = np.maximum(0, self.input_tensor)
        return self.ip1

    def backward(self, error_tensor):
        arr = np.where(self.input_tensor <= 0, 0, error_tensor)
        return arr
