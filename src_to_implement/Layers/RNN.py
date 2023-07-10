import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = None
        self.memorize = False
        self.hidden_gradient = np.zeros((1, self.hidden_size))
        self.tanH_activation = TanH()
        self.sigmoid_activation = Sigmoid()
        self.hidden_state = np.zeros((1, self.hidden_size))
        self.tanH_activation_stack = []
        self.sigmoid_activation_stack = []
        self.fc1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc2 = FullyConnected(self.hidden_size, self.output_size)
        self.gradient_weights = np.zeros_like(self.fc1.gradient_weights)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_rows = self.input_tensor.shape[0]
        self.output = np.zeros((self.input_rows, self.output_size))
        self.x_tilda_stack = []
        if self.memorize is False:
            self.hidden_state = np.zeros((1, self.hidden_size))
        for i in range(0, len(self.input_tensor)):
            self.input_stack = np.hstack((self.hidden_state, self.input_tensor[i][None]))
            self.x_tilda_stack.append(self.input_stack)
            self.fc1_output = self.fc1.forward(self.input_stack)
            self.hidden_state = self.tanH_activation.forward(self.fc1_output)
            self.tanH_activation_stack.append(self.hidden_state)
            self.fc2_output = self.fc2.forward(self.hidden_state)
            self.output[i] = self.sigmoid_activation.forward(self.fc2_output)
            self.sigmoid_activation_stack.append(self.output[i])
        return self.output

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        self.err_tensor_rows = self.error_tensor.shape[0]
        self.gradient_weights = np.zeros_like(self.fc1.weights)
        self.output = np.zeros((self.err_tensor_rows, self.input_size))
        if self.memorize is False:
            self.hidden_gradients = np.zeros((1, self.hidden_size))
        for i, err_val in enumerate(reversed(error_tensor)):
            self.sigmoid_activation.activations = self.sigmoid_activation_stack[-i - 1]
            self.fc2.input_tensor = self.tanH_activation_stack[-i - 1]
            self.tanH_activation.activations = self.tanH_activation_stack[-i - 1]
            self.fc1.input_tensor = self.x_tilda_stack[-i - 1]
            self.sigmoid_gradient = self.sigmoid_activation.backward(err_val)
            self.fc2_gradient = self.fc2.backward(self.sigmoid_gradient[None]) + self.hidden_gradients
            self.tanH_gradient = self.tanH_activation.backward(self.fc2_gradient)
            self.fc1_gradient = self.fc1.backward(self.tanH_gradient)
            self.gradient_weights = self.gradient_weights + self.fc1.gradient_weights
            self.output[-i - 1] = self.fc1_gradient[:, self.hidden_size:]
            self.hidden_gradients = self.fc1_gradient[:, :self.hidden_size]
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.fc2.weights = self.optimizer.calculate_update(self.fc2.weights, self.fc2.gradient_weights)
        return self.output

    def initialize(self, weights_initializer, bias_initializer):
        self.fc1.initialize(weights_initializer, bias_initializer)
        self.fc2.initialize(weights_initializer, bias_initializer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val

    @property
    def weights(self):
        return self.fc1.weights

    @weights.setter
    def weights(self, val):
        self.fc1.weights = val

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, val):
        self._memorize = val

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val
