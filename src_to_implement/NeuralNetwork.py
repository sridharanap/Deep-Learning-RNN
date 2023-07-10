import numpy as np
import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        input_tensor = self.input_tensor
        regularization_loss = 0
        for i, layer in enumerate(self.layers):   # passing through all layers of network
            input_tensor = layer.forward(input_tensor)
            if layer.trainable and layer.optimizer and layer.optimizer.regularizer:
                regularization_loss = regularization_loss + layer.optimizer.regularizer.norm(layer.weights)
        self.new_input_tensor = self.loss_layer.forward(input_tensor, self.label_tensor) + regularization_loss
        return self.new_input_tensor

    def backward(self):
        self.error_tensor = self.loss_layer.backward(self.label_tensor)
        for i in reversed(range(len(self.layers))):
            error_tensor_arr = self.layers[i].backward(self.error_tensor)
            self.error_tensor = error_tensor_arr
        return self.error_tensor

    def append_layer(self, layer):
        self.layer = layer
        if self.layer.trainable:
            self.layer.optimizer = copy.deepcopy(self.optimizer)
            self.layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(self.layer)

    def train(self, iterations):
        self.phase = False
        self.iterations = iterations
        for i in range(self.iterations):
            f = self.forward()
            self.backward()
            self.loss.append(f)

    def test(self, input_tensor):
        self.phase = True
        self.inp_ten = input_tensor
        for i in range(len(self.layers)):
            inp_arr = self.layers[i].forward(self.inp_ten)
            self.inp_ten = inp_arr
        return self.inp_ten

    @property
    def phase(self):
        return self.layers[0].testing_phase

    @phase.setter
    def phase(self, value):
        for i in range(0, len(self.layers)):
            self.layers[i].testing_phase = value
