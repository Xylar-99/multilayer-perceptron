"""The small, from-scratch binary multilayer perceptron used by the project."""

import json
from pathlib import Path

import numpy as np


def relu(values):
    """Apply the ReLU activation: ``max(0, values)``."""
    return np.maximum(0, values)



def binary_cross_entropy(y_true, y_pred):
    """Return average binary cross-entropy for 0/1 labels and probabilities."""
    predictions = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(predictions) + (1 - y_true) * np.log(1 - predictions))


def binary_labels(labels):
    """Convert the project's ``M``/``B`` labels, or existing 0/1 labels, to floats."""
    labels = np.asarray(labels)
    if labels.dtype.kind in {"U", "S", "O"}:
        text_labels = np.char.strip(labels.astype(str))
        if set(np.unique(text_labels)).issubset({"M", "B"}):
            return (text_labels == "M").astype(float)
        try:
            labels = text_labels.astype(float)
        except ValueError as error:
            raise ValueError("Labels must be M/B or binary 0/1 values") from error

    labels = labels.astype(float)
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("Labels must be M/B or binary 0/1 values")
    return labels


def classification_metrics(y_true, predictions):
    """Calculate accuracy, precision, recall, and a binary confusion matrix."""
    true_positive = np.sum((y_true == 1) & (predictions == 1))
    true_negative = np.sum((y_true == 0) & (predictions == 0))
    false_positive = np.sum((y_true == 0) & (predictions == 1))
    false_negative = np.sum((y_true == 1) & (predictions == 0))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "accuracy": np.mean(predictions == y_true),
        "precision": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
        "confusion_matrix": np.array(
            [[true_negative, false_positive], [false_negative, true_positive]]
        ),
    }


class DenseLayer:
    """One fully connected layer with its weights, biases, activations, and gradients."""

    def __init__(self, units, activation):
        """Create a layer with ``units`` neurons and the given activation function."""
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
        """Initialise weights with He/ReLU or Xavier/sigmoid scaling."""
        

        random_generator = rng if rng is not None else np.random.default_rng()
        standard_deviation = np.sqrt(2 / input_features) if self.activation_name == "relu" else np.sqrt(1 / input_features)
        self.input_features = input_features
        self.W = random_generator.normal(0, standard_deviation, (input_features, self.units))
        self.b = np.zeros((1, self.units))


    def forward(self, inputs):
        """Compute ``Z = inputs @ W + b`` and apply this layer's activation."""
        self.inputs = np.asarray(inputs, dtype=float)
        self.Z = self.inputs @ self.W + self.b
        self.A = relu(self.Z) if self.activation_name == "relu" else sigmoid(self.Z)
        return self.A

    def backward(self, gradient, output_layer=False):
        """Calculate ``dW`` and ``db``, then return the gradient for the prior layer."""
        gradient = np.asarray(gradient, dtype=float)
        if output_layer:
            dZ = gradient  # Sigmoid + binary cross-entropy simplifies to prediction - target.
        elif self.activation_name == "relu":
            dZ = gradient * (self.Z > 0)
        else:
            dZ = gradient * self.A * (1 - self.A)

        batch_size = self.inputs.shape[0]
        self.dW = self.inputs.T @ dZ / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        return dZ @ self.W.T

    def update(self, learning_rate):
        """Apply one gradient-descent update to this layer's parameters."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    @classmethod
    def from_saved_data(cls, saved_layer):
        """Create one initialized layer from the JSON representation of a model."""
        weights = np.asarray(saved_layer["weights"], dtype=float)
        layer = cls(saved_layer["units"], saved_layer["activation"])
        layer.input_features = weights.shape[0]
        layer.W = weights
        layer.b = np.asarray(saved_layer["biases"], dtype=float)
        return layer


class MultilayerPerceptron:
    """A sequential binary classifier made from ``DenseLayer`` objects."""

    def __init__(self, layers=None, seed=42):
        self.layers = layers or []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.history = {"loss": [], "accuracy": []}
        self.scaler = None

    def add(self, layer):
        """Append a layer and return this model for readable construction."""
        self.layers.append(layer)
        return self

    def fit(self, X, y, epochs, batch_size, learning_rate):
        """Train with shuffled mini-batches and gradient descent."""
        features = np.asarray(X, dtype=float)
        labels = binary_labels(y).reshape(-1, 1)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("X must be a non-empty two-dimensional feature array")
        if len(features) != len(labels):
            raise ValueError("X and y must contain the same number of examples")
        if not self.layers or self.layers[-1].units != 1 or self.layers[-1].activation_name != "sigmoid":
            raise ValueError("A binary MLP needs one sigmoid output neuron before training")
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be greater than zero")

        batch_rng = np.random.default_rng(self.seed)
        self.history = {"loss": [], "accuracy": []}
        for _ in range(epochs):
            shuffled_indices = batch_rng.permutation(len(features))
            for start_index in range(0, len(features), batch_size):
                batch_indices = shuffled_indices[start_index : start_index + batch_size]
                batch_X = features[batch_indices]
                batch_y = labels[batch_indices]
                predictions = self.forward(batch_X)
                self.backward(batch_y, predictions)
                for layer in self.layers:
                    layer.update(learning_rate)

            metrics = self.evaluate(features, labels)
            self.history["loss"].append(metrics["loss"])
            self.history["accuracy"].append(metrics["accuracy"])
        return self

    def forward(self, X):
        """Propagate feature rows through every layer in order."""
        layer_output = np.asarray(X, dtype=float)
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        return layer_output

    def backward(self, y_true, y_pred):
        """Backpropagate the binary cross-entropy error through every layer."""
        gradient = self.layers[-1].backward(y_pred - y_true, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            gradient = layer.backward(gradient)
        return gradient

    def predict_proba(self, X):
        """Return one malignant-class probability per feature row."""
        return self.forward(X)

    def predict(self, X):
        """Return class ``1`` for probabilities at least 0.5, otherwise ``0``."""
        probabilities = self.predict_proba(X).reshape(-1)
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, X, y):
        """Compute binary loss and classification metrics for labeled data."""
        features = np.asarray(X, dtype=float)
        y_true = binary_labels(y).astype(int).reshape(-1)
        if len(features) != len(y_true):
            raise ValueError("X and y must contain the same number of examples")

        probabilities = self.predict_proba(features).reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(y_true, predictions)
        return {"loss": binary_cross_entropy(y_true, probabilities), **metrics}

    def save(self, filepath):
        """Save topology, parameters, and the attached train-set scaler as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "seed": self.seed,
            "layers": [
                {
                    "units": layer.units,
                    "activation": layer.activation_name,
                    "weights": layer.W.tolist(),
                    "biases": layer.b.tolist(),
                }
                for layer in self.layers
            ],
            "scaler": self.scaler,
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(model_data, file, indent=4)

    @classmethod
    def load(cls, filepath):
        """Restore a model saved by :meth:`save`."""
        with Path(filepath).open("r", encoding="utf-8") as file:
            model_data = json.load(file)

        model = cls(seed=model_data.get("seed", 42))
        model.layers = [DenseLayer.from_saved_data(layer) for layer in model_data["layers"]]
        model.scaler = model_data.get("scaler")
        return model
