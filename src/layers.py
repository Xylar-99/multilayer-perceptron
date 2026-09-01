import numpy as np

class Dense:
    def __init__(self, units, input_shape=None, activation=None, weights_initializer='heUniform'):
        pass
    def initialize(self, input_dim):
        pass
    def forward(self, X):
        pass
    def backward(self, dA):
        pass
    def update(self, optimizer):
        pass

class Activation:
    @staticmethod
    def forward(Z, activation):
        pass
    @staticmethod
    def backward(dA, A_prev, activation):
        pass
