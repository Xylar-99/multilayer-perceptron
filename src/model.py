import json
from pathlib import Path
import numpy as np
import sys




def binary_labels(labels):
    """Convert B/M labels to one-hot encoded labels."""
    labels = np.asarray(labels).reshape(-1)

    y = np.zeros((len(labels), 2), dtype=int)

    y[labels == "B", 0] = 1
    y[labels == "M", 1] = 1

    return y



class BCE:
    """Return the average binary cross-entropy loss."""
    @staticmethod
    def compute(targets, predictions):
        predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        return -np.mean(
            targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
        )

    @staticmethod
    def compute_gradient(targets, predictions):
        predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        return -(targets / predictions) + (1 - targets) / (1 - predictions)



class CCE:
    """Return the average categorical cross-entropy loss."""

    @staticmethod
    def compute(targets, predictions):
        predictions = np.clip(predictions, 1e-12, 1.0)

        return -np.mean(
            np.sum(targets * np.log(predictions), axis=1)
        )

    @staticmethod
    def compute_gradient(targets, predictions):
        predictions = np.clip(predictions, 1e-12, 1.0)
        return -(targets / predictions)
    

class ReLU:

    @staticmethod
    def forward(inputs):
        return np.maximum(0.0, inputs)

    @staticmethod
    def backward(dA, Z):
        return dA * (Z > 0)


class Softmax:
    """The softmax activation: exp(x) / sum(exp(x))."""

    @staticmethod
    def forward(inputs):
        inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(inputs)

        return exp_values / np.sum(exp_values,axis=1,keepdims=True)


class DenseLayer:

    def __init__(self, input_features, units, activation="relu", seed=42):
        self.activation = activation
        self.weights = np.random.default_rng(seed).normal(0, 1 / np.sqrt(input_features),(input_features, units))
        self.biases = np.zeros((1, units))

        self.inputs = None
        self.Z = None
        self.A = None
        self.dZ = None
        self.dW = None
        self.db = None

    def forward(self, inputs):
        self.inputs = np.asarray(inputs, dtype=float)
        self.Z = self.inputs @ self.weights + self.biases

        if self.activation == "relu":
            self.A = ReLU.forward(self.Z)
        elif self.activation == "softmax":
            self.A = Softmax.forward(self.Z)
        else:
            raise ValueError("Unsupported activation")

        return self.A

    def backward(self, dA):
        if self.activation == "relu":
            self.dZ = ReLU.backward(dA, self.Z)
        else:
            self.dZ = dA

        batch_size = self.inputs.shape[0]

        self.dW = (self.inputs.T @ self.dZ) / batch_size
        self.db = np.sum(self.dZ, axis=0, keepdims=True) / batch_size

        return self.dZ @ self.weights.T

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db




class MultilayerPerceptron:
    """A binary MLP made from Dense, ReLU, and Softmax layers."""

    def __init__(self, input_features, hidden_layers=(24, 24), epochs=100, batch_size=32, learning_rate=0.01, seed=42):
        self.input_features = input_features
        self.hidden_layers = list(hidden_layers) + [2]
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        self.history = {"loss": [], "accuracy": [] , "val_loss": [], "val_accuracy": []}
        self.scaler = None
        self.layers = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        for layer_index, units in enumerate(self.hidden_layers):
            activation = ("softmax" if layer_index == len(self.hidden_layers) - 1 else "relu")
            layer = DenseLayer(input_features, units, activation)
            self.layers.append(layer)
            input_features = units

    def forward(self, X):
        inputs = np.asarray(X, dtype=float)

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, predictions, targets):
        dZ = predictions - targets

        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    # def fit(self, X, y, epochs, batch_size, learning_rate):
    #     X = np.asarray(X, dtype=float)
    #     y = binary_labels(y)

    #     for epoch in range(epochs):
    #         indices = self.rng.permutation(len(X))
    #         X_shuffled = X[indices]
    #         y_shuffled = y[indices]

    #         for start in range(0, len(X), batch_size):
    #             end = start + batch_size

    #             X_batch = X_shuffled[start:end]
    #             y_batch = y_shuffled[start:end]

    #             predictions = self.forward(X_batch)

    #             self.backward(predictions, y_batch)
    #             self.update(learning_rate)


    def fit(self, X_train, y_train, X_valid, y_valid):

        X_train = np.asarray(X_train, dtype=float)
        X_valid = np.asarray(X_valid, dtype=float)

        y_train = binary_labels(y_train)
        y_valid = binary_labels(y_valid)

        batch_size = self.batch_size
        learning_rate = self.learning_rate
        epochs = self.epochs

        for epoch in range(epochs):
            indices = self.rng.permutation(len(X_train))

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, len(X_train), batch_size):
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                predictions = self.forward(X_batch)

                self.backward(predictions, y_batch)
                self.update(learning_rate)


            # Compute and store training and validation loss and accuracy
            train_predictions = self.forward(X_train)
            valid_predictions = self.forward(X_valid)

            train_loss = CCE.compute(y_train,train_predictions)
            valid_loss = CCE.compute(y_valid, valid_predictions)

            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(valid_loss)
            self.history["accuracy"].append(np.mean(np.argmax(train_predictions, axis=1) == np.argmax(y_train, axis=1)))
            self.history["val_accuracy"].append(np.mean(np.argmax(valid_predictions, axis=1) == np.argmax(y_valid, axis=1)))

            print(
                f"epoch {epoch + 1:02d}/{epochs} "
                f"- loss: {train_loss:.4f} "
                f"- val_loss: {valid_loss:.4f}"
            )

    def predict_proba(self, X):
        """Return malignant probability for each sample."""
        X = np.asarray(X, dtype=float)

        predictions = self.forward(X)

        return predictions[:, 1]
    
    def predict(self, X):
        """Return B for benign and M for malignant."""
        X = np.asarray(X, dtype=float)

        predictions = self.forward(X)
        classes = np.argmax(predictions, axis=1)

        return np.where(classes == 0, "B", "M")



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
        print(f"Model saved to {filepath}")

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
