import numpy as np


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        self.weights = weights
        sub_gradient = self.alpha * np.sign(self.weights)
        return sub_gradient

    def norm(self, weights):
        self.weights = weights
        regularization_loss = self.alpha * np.sum(np.abs(self.weights))
        return regularization_loss


class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        sub_gradient = self.alpha * weights
        return sub_gradient

    def norm(self, weights):
        weight_l2 = np.square(np.linalg.norm(weights))
        regularization_loss = self.alpha * weight_l2
        return regularization_loss
