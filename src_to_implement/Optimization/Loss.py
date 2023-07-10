import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor=None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor= prediction_tensor
        self.label_tensor= label_tensor
        self.loss=-np.log(self.prediction_tensor + np.finfo(float).eps)
        self.cross_loss= np.sum(np.multiply(label_tensor, self.loss))
        return self.cross_loss

    def backward(self, label_tensor):
        self.label_tensor= label_tensor
        self.en=-self.label_tensor/(self.prediction_tensor + np.finfo(float).eps)

        return self.en