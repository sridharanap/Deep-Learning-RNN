import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.func = np.random.random(self.input_tensor.shape)
        self.drop = self.func < self.probability
        self.dropout_tensor = self.input_tensor * self.drop * (1 / self.probability)
        if self.testing_phase == True:
            self.dropout_tensor = self.input_tensor
        return self.dropout_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.dropout_error = self.error_tensor * self.drop * (1 / self.probability)
        if self.testing_phase == True:
            self.dropout_error = self.error_tensor
        return self.dropout_error
