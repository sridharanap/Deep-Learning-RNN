import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.bias = None
        self.weights = None
        self.optimizer = None
        self.initialize(None,None)
        self.test_mean = None
        self.test_var = None
        self.batch_mean = 0.
        self.batch_var = 0.

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):
        self.tensor = tensor
        if self.tensor.ndim == 4:
            interim_shape1 = self.tensor.shape[:2]
            interim_shape2 = self.tensor.reshape(*interim_shape1, -1)
            return np.swapaxes(interim_shape2, 1, 2).reshape(-1, self.tensor.shape[1])
        elif self.tensor.ndim == 2:
            interim_shape1 = self.input_shape[0]
            interim_shape2 = self.input_shape[1]
            output_reformed_tensor = self.tensor.reshape(interim_shape1, -1, interim_shape2)
            return np.swapaxes(output_reformed_tensor, 1, 2).reshape(*self.input_shape)
        else:
            return self.tensor

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        self.input_tensor = self.reformat(input_tensor)
        self.batch_mean = self.input_tensor.mean(axis=0, keepdims=True)
        self.batch_var = self.input_tensor.var(axis=0, keepdims=True)
        self.alpha = 0.8
        if self.testing_phase is True:
            self.batch_mean = self.test_mean
            self.batch_var = self.test_var
        else:
            if self.test_mean is None and self.test_var is None:
                self.test_mean = self.batch_mean
                self.test_var = self.batch_var
            else:
                self.alt_mean = self.alpha * self.test_mean + (1 - self.alpha) * self.batch_mean
                self.test_mean = self.alt_mean
                self.alt_var = self.alpha * self.test_var + (1 - self.alpha) * self.batch_var
                self.test_var = self.alt_var
        delta = self.input_tensor - self.batch_mean
        self.norm_input = delta / np.sqrt(self.batch_var + np.finfo(float).eps)
        self.y_t = self.norm_input * self.weights + self.bias
        self.output = self.reformat(self.y_t)
        return self.output

    def backward(self, error_tensor):
        self.reformed_error = self.reformat(error_tensor)
        self.gradient_weights = np.sum(self.reformed_error * self.norm_input, axis=0)
        self.gradient_bias = np.sum(self.reformed_error, axis=0)
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        self.gradient_error = compute_bn_gradients(self.reformed_error, self.input_tensor, self.weights,
                                                   self.batch_mean, self.batch_var)
        self.gradient_tensor = self.reformat(self.gradient_error)
        return self.gradient_tensor

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, val):
        self._gradient_bias = val

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val
