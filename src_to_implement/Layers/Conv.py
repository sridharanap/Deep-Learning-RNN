import copy
import math
import numpy as np
from scipy.signal import convolve, correlate
from Layers.Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super(Conv, self).__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.chnls = convolution_shape[0]
        self.weights = np.random.uniform(0.0, 1.0, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0.0, 1.0, num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self.optimizer = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = self.input_tensor.shape[0]
        self.output_tensor = np.empty((batch_size, self.num_kernels, *input_tensor.shape[2:]))
        for i, inp in enumerate(input_tensor):
            for j, k in enumerate(self.weights):
                correlation = correlate(inp, k, mode='same')
                image_size = math.floor(self.chnls / 2)
                updated_correlation = correlation[image_size]
                self.output_tensor[i][j] = updated_correlation + self.bias[j]
        if len(self.convolution_shape) == 2:
            self.output_stride = self.output_tensor[::, ::, ::self.stride_shape[0]]
        else:
            self.output_stride = self.output_tensor[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]]
        return self.output_stride

    def backward(self, error_tensor):
        gradient_weights = np.zeros(self.weights.shape)
        gradient_bias = np.zeros(self.bias.shape)
        resampled_error = np.zeros((*self.output_tensor.shape[0:2], *self.output_tensor.shape[2:]))
        if len(self.convolution_shape) == 2:
            resampled_error[::, ::, ::self.stride_shape[0]] = error_tensor
        else:
            resampled_error[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        updated_kernels = np.swapaxes(self.weights, 1, 0)
        updated_kernels = np.flip(updated_kernels, axis=1)
        output_tensor = np.zeros_like(self.input_tensor)
        for i, error in enumerate(resampled_error):
            for j, channel in enumerate(updated_kernels):
                convol = convolve(error, channel, mode='same')
                img_size = math.floor(self.num_kernels /2)
                output_tensor[i][j] = convol[img_size]
        if len(self.convolution_shape) == 2:
            input_pad = self.input_tensor
        else:
            input_pad = np.pad(self.input_tensor, ( (0, 0), (0, 0), (math.floor((self.convolution_shape[1] - 1) /2), math.floor(self.convolution_shape[1] / 2)),
                (math.floor((self.convolution_shape[2] - 1) / 2), math.floor(self.convolution_shape[2] / 2))))
        for i, error in enumerate(resampled_error):   # Calculating gradient wrt weights and bias.
            for j, err_channel in enumerate(error):
                for k in range(self.chnls):
                    gradient_weights[j][k] += correlate(input_pad[i][k], err_channel, mode='valid')
                gradient_bias[j] += np.sum(err_channel)

        self.gradient_weights = gradient_weights
        self.gradient_bias = gradient_bias
        if self.optimizer:   # Updating the values for weights and bias for the next step
            self.weights = self._weights_optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.weights.shape[1:]), self.weights.shape[0] * np.prod(self.weights.shape[2:]))
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.bias.shape), np.prod(self.bias.shape))

    @property
    def optimizer(self):
        return self._weights_optimizer


    @optimizer.setter
    def optimizer(self, val):
        self._weights_optimizer = copy.deepcopy(val)
        self._bias_optimizer = copy.deepcopy(val)

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