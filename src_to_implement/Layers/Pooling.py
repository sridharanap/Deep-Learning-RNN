import numpy as np
from numpy.core.fromnumeric import shape
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.pooling_shape = pooling_shape
        self.stride_shape = stride_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        x = int((input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0]) + 1     # Padding
        y = int((input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1]) + 1     # Padding
        output_tensor = np.zeros((input_tensor.shape[0], input_tensor.shape[1], x, y))
        for b in range(output_tensor.shape[0]):
            for c in range(output_tensor.shape[1]):
                val_c = 0
                for i in range(output_tensor.shape[2]):
                    val_i = 0
                    for j in range(output_tensor.shape[3]):
                        output_tensor[b, c, i, j] = np.max(
                            input_tensor[b, c, val_c:val_c + self.pooling_shape[0], val_i:val_i + self.pooling_shape[1]])
                        val_i = val_i + self.stride_shape[1]
                    val_c = val_c + self.stride_shape[0]
        return output_tensor

    def backward(self, error_tensor):
        err_output = np.zeros_like(self.input_tensor)
        for b in range(error_tensor.shape[0]):
            for c in range(error_tensor.shape[1]):
                val_c = 0
                for i in range(error_tensor.shape[2]):
                    val_i = 0
                    for j in range(error_tensor.shape[3]):
                        pool = self.input_tensor[b, c, val_c:val_c + self.pooling_shape[0],
                               val_i:val_i + self.pooling_shape[1]]
                        err_output[b, c, val_c:val_c + self.pooling_shape[0], val_i:val_i + self.pooling_shape[1]][
                            pool == np.max(pool)] += error_tensor[b, c, i, j]
                        val_i = val_i + self.stride_shape[1]
                    val_c = val_c + self.stride_shape[0]
        return err_output
