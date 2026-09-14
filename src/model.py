"""A small binary multilayer perceptron built with NumPy."""

import json
from pathlib import Path

import numpy as np


def binary_labels(labels):
    """Convert B/M labels, or existing 0/1 labels, to integers."""
    labels = np.asarray(labels).reshape(-1)

    if labels.dtype.kind in {"U", "S", "O"}:
        labels = np.char.strip(labels.astype(str))
        if set(np.unique(labels)).issubset({"B", "M"}):
            return (labels == "M").astype(int)

    numeric_labels = labels.astype(float)
    if not np.all(np.isin(numeric_labels, (0.0, 1.0))):
        raise ValueError("Labels must be B/M or binary values 0/1")
    return numeric_labels.astype(int)


def binary_cross_entropy(targets, predictions):
    """Return the average binary cross-entropy loss."""
    predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
    return -np.mean(
        targets * np.log(predictions)
        + (1.0 - targets) * np.log(1.0 - predictions)
    )


def classification_metrics(targets, predictions):
    """Return simple binary classification metrics."""
    true_positive = np.sum((targets == 1) & (predictions == 1))
    true_negative = np.sum((targets == 0) & (predictions == 0))
    false_positive = np.sum((targets == 0) & (predictions == 1))
    false_negative = np.sum((targets == 1) & (predictions == 0))

    precision_total = true_positive + false_positive
    recall_total = true_positive + false_negative
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "precision": true_positive / precision_total if precision_total else 0.0,
        "recall": true_positive / recall_total if recall_total else 0.0,
        "confusion_matrix": np.array(
            [[true_negative, false_positive], [false_negative, true_positive]]
        ),
    }


class ReLU:
    """The ReLU activation: ``max(0, x)``."""

    def __init__(self):
        self.inputs = None
    @staticmethod
    def forward(self, values):
        self.inputs = np.asarray(values, dtype=float)
        return np.maximum(0.0, self.inputs)

    @staticmethod
    def backward(self, gradient):
        return gradient * (self.inputs > 0.0)


class Sigmoid:
    """The sigmoid activation used by the final binary output."""

    def __init__(self):
        self.outputs = None

    @staticmethod
    def forward(self, values):
        values = np.asarray(values, dtype=float)
        self.outputs = 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))
        return self.outputs
    @staticmethod
    def backward(self, gradient):
        return gradient * self.outputs * (1.0 - self.outputs)


class DenseLayer:
    """A fully connected linear layer: ``Z = inputs @ weights + biases``."""

    def __init__(self, input_features, units , activation="relu"):
        self.input_features = input_features
        self.units = units

        self.activation = activation
        self.weights = np.random.default_rng().normal(0.0, np.sqrt(2.0 / input_features), (input_features, units))
        self.biases = np.zeros((1, units))
        self.inputs = None
        self.dW = None
        self.db = None
        self.dA = None
        self.dZ = None

    def forward(self, inputs):
        self.inputs = np.asarray(inputs, dtype=float)
        if self.activation == "relu":
            self.dA = ReLU.forward(self, self.inputs @ self.weights + self.biases)
        elif self.activation == "softmax":
            self.dA = Softmax.forward(self, self.inputs @ self.weights + self.biases)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        return self.dA

    def backward(self, dZ):
        """Calculate parameter gradients and return the previous gradient."""
        self.dZ = np.asarray(dZ, dtype=float)
        self.dW = self.inputs.T @ self.dZ / len(self.inputs)
        self.db = np.sum(self.dZ, axis=0, keepdims=True) / len(self.inputs)
        return self.dZ @ self.weights.T

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db


class MultilayerPerceptron:
    """A binary MLP made from Dense, ReLU, and Sigmoid layers."""

    def __init__(self, input_features, hidden_layers=(6, 8), seed=42):
        self.input_features = input_features
        self.hidden_layers = list(hidden_layers)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.history = {"loss": [], "accuracy": []}
        self.scaler = None
        self.layers = []

        previous_features = input_features
        for units in hidden_layers:
            self.layers.append(DenseLayer(previous_features, units, self.rng))
            self.layers.append(ReLU())
            previous_features = units

        self.layers.append(DenseLayer(previous_features, 1, self.rng))
        self.layers.append(Sigmoid())

    def forward(self, X):
        """Return the final output of the network."""
        inputs = np.asarray(X, dtype=float)
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, dA):
        """Backpropagate the loss gradient through the network."""
        gradient = np.asarray(dA, dtype=float)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def update(self, learning_rate):
        """Update the weights and biases of each layer."""
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.update(learning_rate)
            
    def fit(self, X, y, epochs, batch_size, learning_rate):
        """Train with mini-batch gradient descent."""
        X = np.asarray(X, dtype=float)
        y = binary_labels(y)

        for epoch in range(epochs):
            indices = self.rng.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, len(X), batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                predictions = self.forward(X_batch)
                loss_gradient = predictions - y_batch.reshape(-1, 1)
                self.backward(loss_gradient)
                self.update(learning_rate)

            epoch_loss = binary_cross_entropy(y, self.predict_proba(X))
            epoch_accuracy = np.mean(self.predict(X) == y)
            self.history["loss"].append(epoch_loss)
            self.history["accuracy"].append(epoch_accuracy)


    def predict_proba(self, X):
        """Return one malignant probability for each input row."""
        return self.forward(X).reshape(-1)

    def predict(self, X):
        """Return 0 for benign and 1 for malignant."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def evaluate(self, X, y):
        """Return loss, accuracy, precision, recall, and a confusion matrix."""
        targets = binary_labels(y)
        probabilities = self.predict_proba(X)
        if len(probabilities) != len(targets):
            raise ValueError("X and y must contain the same number of examples")

        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(targets, predictions)
        return {
            "loss": float(binary_cross_entropy(targets, probabilities)),
            **metrics,
        }

    def save(self, filepath):
        """Save the network architecture and learned parameters as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "input_features": self.input_features,
            "hidden_layers": self.hidden_layers,
            "seed": self.seed,
            "layers": [
                {"weights": layer.weights.tolist(), "biases": layer.biases.tolist()}
                for layer in self.layers
                if isinstance(layer, DenseLayer)
            ],
            "scaler": self.scaler,
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(model_data, file, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load a model saved with :meth:`save`."""
        with Path(filepath).open(encoding="utf-8") as file:
            model_data = json.load(file)

        model = cls(
            model_data["input_features"],
            model_data["hidden_layers"],
            model_data.get("seed", 42),
        )
        dense_layers = [layer for layer in model.layers if isinstance(layer, DenseLayer)]

        for layer, saved_layer in zip(dense_layers, model_data["layers"]):
            layer.weights = np.asarray(saved_layer["weights"], dtype=float)
            layer.biases = np.asarray(saved_layer["biases"], dtype=float)

        model.scaler = model_data.get("scaler")
        return model
