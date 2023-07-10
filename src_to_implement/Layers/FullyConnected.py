from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.optimizer = None
        self.gradient_weights = None
        self.weights = np.random.rand(input_size +1, output_size)  # extra row to add bias for weights

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = len(self.input_tensor)
        self.input= np.dot(self.input_tensor, self.weights)
        return self.input

    @ property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.output_tensor = np.dot(self.error_tensor, np.transpose(self.weights[:-1]))
        self.gradient_weights = np.dot(np.transpose(self.input_tensor), self.error_tensor)
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return self.output_tensor

    @ property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val2):
        self._gradient_weights = val2

    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1] = weights_initializer.initialize(self.weights[:-1].shape, self.weights[:-1].shape[0], self.weights[:-1].shape[1])
        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape, 1, self.weights[-1].shape[0])

    @property
    def input_tensor(self):
        return self._input_tensor

    @input_tensor.setter
    def input_tensor(self, value):
        if value is None:
            self._input_tensor = None
        else:
            self._input_tensor = np.c_[value, np.ones(value.shape[0])]