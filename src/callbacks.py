import numpy as np

class History:
    def __init__(self):
        pass
    def append(self, metrics):
        pass
    def get(self, key):
        pass

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        pass
    def check(self, current_value, epoch):
        pass
