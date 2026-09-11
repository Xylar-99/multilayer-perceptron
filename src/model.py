import numpy as np
import json
import os




class Sigmoid:
    """Sigmoid activation: sigma(z) = 1 / (1 + e^(-z))"""
    @staticmethod
    def forward(Z):
        """Compute sigmoid without overflowing for large values."""
        Z = np.asarray(Z, dtype=float)
        result = np.empty_like(Z)

        positive = Z >= 0
        result[positive] = 1 / (1 + np.exp(-Z[positive]))

        exp_Z = np.exp(Z[~positive])
        result[~positive] = exp_Z / (1 + exp_Z)

        return result

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
        """Initialize small weights and zero biases."""
        self.input_features = input_features

        rng = rng or np.random.default_rng()
        if self.activation_name == "relu":
            standard_deviation = np.sqrt(2 / input_features)
        else:
            standard_deviation = np.sqrt(1 / input_features)

        self.W = rng.normal(
            0,
            standard_deviation,
            (self.input_features, self.units),
        )
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


    def backward(self, gradient, output_layer=False):
        """Calculate gradients and return the gradient for the previous layer."""

        # The output gradient from sigmoid + binary cross-entropy is already dZ.
        if output_layer:
            dZ = gradient
        elif self.activation_name == "relu":
            dZ = ReLU.backward(gradient, Z=self.Z, A=self.A)
        elif self.activation_name == "softmax":
            dZ = Softmax.backward(gradient, Z=self.Z, A=self.A)
        elif self.activation_name == "sigmoid":
            dZ = Sigmoid.backward(gradient, Z=self.Z, A=self.A)
        elif self.activation_name == "tanh":
            dZ = Tanh.backward(gradient, Z=self.Z, A=self.A)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        batch_size = self.inputs.shape[0]
        self.dW = np.dot(self.inputs.T, dZ) / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        return np.dot(dZ, self.W.T)


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
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )


    @staticmethod
    def gradient(y_true, y_pred, eps=1e-15):
        """Compute loss derivative with respect to prediction."""
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))



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
        self.history = {"loss": [], "accuracy": []}
        self.scaler = None

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
        dA = self.layers[-1].backward(
            y_pred - y_true,
            output_layer=True,
        )
        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA)
        return dA


    def fit(self, X, y, epochs, batch_size, learning_rate):
        """Train the neural network using mini-batch gradient descent."""
        if len(self.layers) == 0:
            raise ValueError("Add at least one layer before training")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of examples")

        # Convert text labels to 0/1 and make labels column-shaped.
        if y.dtype.kind in {"U", "S", "O"}:
            y = (y == "M").astype(float)
        y = y.astype(float).reshape(-1, 1)

        number_of_samples = len(X)

        np.random.seed(self.seed)
        self.history = {"loss": [], "accuracy": []}

        # Train the network several times over the whole dataset.
        for epoch in range(epochs):
            indexes = np.arange(number_of_samples)
            np.random.shuffle(indexes)

            for start in range(0, number_of_samples, batch_size):
                end = start + batch_size
                batch_indexes = indexes[start:end]

                batch_X = X[batch_indexes]
                batch_y = y[batch_indexes]

                predictions = self.forward(batch_X)
                self.backward(batch_y, predictions)

                for layer in self.layers:
                    layer.W -= learning_rate * layer.dW
                    layer.b -= learning_rate * layer.db

            results = self.evaluate(X, y)
            self.history["loss"].append(results["loss"])
            self.history["accuracy"].append(results["accuracy"])

        return self



        


    def predict_proba(self, X):
        """Return predicted probability distribution for input X."""
        X = np.asarray(X, dtype=float)
        return self.forward(X)
        

    def predict(self, X):
        """Return binary class prediction (0 or 1) for input X."""
        probabilities = self.predict_proba(X).reshape(-1)
        return (probabilities >= 0.5).astype(int)


    def evaluate(self, X, y):
        """Compute BCE loss, accuracy, precision, recall, and confusion matrix."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Convert text labels to 0/1.
        if y.dtype.kind in {"U", "S", "O"}:
            y = (y == "M").astype(int)
        y_true = y.astype(int).reshape(-1)
        probabilities = self.predict_proba(X).reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)

        true_positive = np.sum((y_true == 1) & (predictions == 1))
        true_negative = np.sum((y_true == 0) & (predictions == 0))
        false_positive = np.sum((y_true == 0) & (predictions == 1))
        false_negative = np.sum((y_true == 1) & (predictions == 0))

        accuracy = np.mean(predictions == y_true)
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0.0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0.0

        return {
            "loss": BinaryCrossEntropy.compute(y_true, probabilities),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": np.array([
                [true_negative, false_positive],
                [false_negative, true_positive],
            ]),
        }


    def save(self, filepath):
        """Save model topology, weights, biases, and scaler to JSON."""
        model_data = {
            "seed": self.seed,
            "layers": [],
            "scaler": getattr(self, "scaler", None),
        }

        for layer in self.layers:
            model_data["layers"].append({
                "units": layer.units,
                "activation": layer.activation_name,
                "weights": layer.W.tolist(),
                "biases": layer.b.tolist(),
            })

        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(model_data, file, indent=4)


    @classmethod
    def load(cls, filepath):
        """Load model topology, weights, biases, and scaler from JSON."""
        with open(filepath, "r", encoding="utf-8") as file:
            model_data = json.load(file)

        model = cls(seed=model_data.get("seed", 42))

        for saved_layer in model_data["layers"]:
            weights = np.array(saved_layer["weights"], dtype=float)
            biases = np.array(saved_layer["biases"], dtype=float)

            layer = DenseLayer(
                saved_layer["units"],
                saved_layer["activation"],
            )
            layer.build(weights.shape[0], rng=model.rng)
            layer.W = weights
            layer.b = biases
            model.add(layer)

        model.scaler = model_data.get("scaler")
        return model
