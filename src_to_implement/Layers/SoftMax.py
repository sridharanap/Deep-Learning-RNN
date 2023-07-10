from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.ip_tensor=None

    def forward(self, input_tensor):
        self.input_tensor= input_tensor
        self.ip_tensor = self.input_tensor - np.max(self.input_tensor)
        self.num = np.exp(self.ip_tensor)
        self.ip = self.num/np.sum(self.num, axis=1, keepdims=True)
        return self.ip

    def backward(self, error_tensor):
        self.error_tensor= error_tensor
        self.sum= np.sum(self.error_tensor * self.ip, axis=1, keepdims=True)
        self.err_tensor= self.ip * (self.error_tensor - self.sum)
        return self.err_tensor