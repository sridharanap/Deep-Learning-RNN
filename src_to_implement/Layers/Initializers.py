import numpy as np


class Constant:
    def __init__(self, c=0.1):
        self.c = c

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.wi = np.full(self.weights_shape, self.c)
        return self.wi


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights_shape=weights_shape
        self.fan_in=fan_in
        self.fan_out=fan_out
        # self.fan_in, self.fan_out = self.weights_shape
        self.wt = np.random.uniform(0.0,1.0,self.weights_shape)
        return self.wt


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):

        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        self.weight_shape=weights_shape
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, self.weight_shape)