import numpy as np
import json
import os
import pandas as pd




class Sigmoid:
    """Sigmoid activation: sigma(z) = 1 / (1 + e^(-z))"""
    @staticmethod
    def forward(Z):
        """Compute sigmoid activation."""
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def backward(dA, Z=None, A=None):
        """Compute gradient: dA * A * (1 - A)"""
        if A is None:
            A = Sigmoid.forward(Z)
        return dA * A * (1 - A)




class Softmax:
    """Softmax activation: e^(z_i) / sum(e^(z_j))"""

    @staticmethod
    def forward(Z):
        """Compute stable softmax probability distribution."""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    @staticmethod
    def backward(dA, Z=None, A=None):
        """Backpropagate gradient through softmax."""

        if A is None:
            if Z is None:
                raise ValueError("Either Z or A must be provided")
            A = Softmax.forward(Z)

        dZ = np.zeros_like(A)

        for i in range(A.shape[0]):  # each sample
            a = A[i]

            # Softmax Jacobian:
            # J = diag(a) - a @ a.T
            J = np.diag(a) - np.outer(a, a)

            # dZ = dA @ J
            dZ[i] = dA[i] @ J

        return dZ
    


class ReLU:
    """ReLU activation: g(z) = max(0, z)"""
    @staticmethod
    def forward(Z):
        """Compute ReLU activation."""
        return np.maximum(0, Z)

    @staticmethod
    def backward(dA, Z=None, A=None):
        """Compute ReLU derivative."""
        if A is None:
            A = ReLU.forward(Z)
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ



class Tanh:
    """Tanh activation: g(z) = (e^z - e^-z) / (e^z + e^-z)"""
    @staticmethod
    def forward(Z):
        """Compute Tanh activation."""
        return np.tanh(Z)

    @staticmethod
    def backward(dA, Z=None, A=None):
        """Compute Tanh derivative."""
        if A is None:
            A = Tanh.forward(Z)
        return dA * (1 - A ** 2)




class DenseLayer:
    """
    Fully-connected layer storing weight matrix W and bias vector b.
    Computes Z = inputs @ W + b, then applies activation A = g(Z).
    """
    def __init__(self, units, activation):
        self.units = units
        self.activation_name = activation
        self.input_features = None

        self.W = None
        self.b = None
        self.dW = None
        self.db = None

        self.inputs = None
        self.Z = None
        self.A = None

    def build(self, input_features, rng=None):
        """Initialize weights W (He/Xavier) and biases b."""
        self.input_features = input_features

        rng = rng or np.random.default_rng()
        self.W = rng.standard_normal((self.input_features, self.units))
        self.b = np.zeros((1, self.units))


    def forward(self, inputs):
        """Forward pass: compute Z = inputs @ W + b, then A = g(Z)."""
        self.inputs = inputs
        self.Z = np.dot(inputs, self.W) + self.b
        if self.activation_name == "relu":
            self.A = ReLU.forward(self.Z)
        elif self.activation_name == "softmax":
            self.A = Softmax.forward(self.Z)
        elif self.activation_name == "sigmoid":
            self.A = Sigmoid.forward(self.Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        return self.A


    def backward(self, grad, is_delta=False):
        """Backward pass: compute dW, db, and return dX for previous layer."""
        
        if self.activation_name == "relu":
            dZ = ReLU.backward(grad, Z=self.Z, A=self.A)
        elif self.activation_name == "softmax":
            dZ = Softmax.backward(grad, Z=self.Z, A=self.A)
        elif self.activation_name == "sigmoid":
            dZ = Sigmoid.backward(grad, Z=self.Z, A=self.A)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")
        



class CategoricalCrossEntropy:
    """Categorical Cross-Entropy Loss."""

    @staticmethod
    def compute(y_true, y_pred):
        """Compute CCE loss value."""

        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        N = y_true.shape[0]

        loss = -np.sum(y_true * np.log(y_pred)) / N

        return loss

    @staticmethod
    def gradient(y_true, y_pred, eps=1e-15):
        """Compute loss derivative with respect to prediction."""

        y_pred = np.clip(y_pred, eps, 1 - eps)

        N = y_true.shape[0]

        return -(y_true / y_pred) / N


class BinaryCrossEntropy:
    """Binary Cross-Entropy Loss: E = - (1/N) * sum [ y*log(p) + (1-y)*log(1-p) ]"""
    @staticmethod
    def compute(y_true, y_pred):
        """Compute BCE loss value."""
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))


    @staticmethod
    def gradient(y_true, y_pred, eps=1e-15):
        """Compute loss derivative with respect to prediction."""
        
        return y_true - y_pred;



class MultilayerPerceptron:
    """
    Multilayer Perceptron (MLP) coordinator.
    Chains DenseLayers, coordinates forward/backward passes, mini-batch training,
    feature normalization, evaluation, and JSON serialization.
    """
    def __init__(self, layers=None, seed=42):
        self.layers = layers or []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    def add(self, layer):
        """Append a DenseLayer to the network."""
        self.layers.append(layer)
        return self
        

    def forward(self, X):
        """Propagate input X sequentially through all layers."""
        
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred):
        """Backpropagate error from output layer to input layer."""
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                grad = CategoricalCrossEntropy.gradient(y_true, y_pred)
            else:
                grad = self.layers[i + 1].backward(grad)
            layer.backward(grad)


    def fit(self, train_data, val_data=None, epochs=84, batch_size=8, learning_rate=0.0314):
        """Train the neural network using mini-batch gradient descent."""

        
        


    def predict_proba(self, X):
        """Return predicted probability distribution for input X."""
        pass
        

    def predict(self, X):
        """Return binary class prediction (0 or 1) for input X."""
        pass

    def evaluate(self, X, y):
        """Compute BCE loss, accuracy, precision, recall, and confusion matrix."""
        pass

    def save(self, filepath):
        """Save model topology, weights, biases, and scaler to JSON."""
        pass

    @classmethod
    def load(cls, filepath):
        """Load model topology, weights, biases, and scaler from JSON."""
        pass
