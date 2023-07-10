import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super(Flatten, self).__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        self.in_tensor = np.reshape(input_tensor, (input_tensor.shape[0], np.prod(input_tensor.shape[1:])))
        return self.in_tensor

    def backward(self, error_tensor):
        self.err_tensor = np.reshape(error_tensor, self.input_shape)
        return self.err_tensor