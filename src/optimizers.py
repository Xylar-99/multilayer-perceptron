import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        pass
    def update(self, W, b, dW, db):
        pass

class SGDWithMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        pass
    def update(self, W, b, dW, db):
        pass

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        pass
    def update(self, W, b, dW, db):
        pass

class RMSprop:
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        pass
    def update(self, W, b, dW, db):
        pass
