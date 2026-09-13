import json
import os

import numpy as np



class Softmax:
    """Softmax activation: e^(z_i) / sum(e^(z_j))."""

    @staticmethod
    def forward(Z):
        """Compute a stable softmax probability distribution for each sample."""
        largest_logit = np.max(Z, axis=1, keepdims=True)
        shifted_logits = Z - largest_logit
        exponentials = np.exp(shifted_logits)
        exponential_sums = np.sum(exponentials, axis=1, keepdims=True)

        return exponentials / exponential_sums

    @staticmethod
    def backward(dA, Z=None, A=None):
        """Return dZ by multiplying dA by each sample's softmax Jacobian."""
        if A is None:
            if Z is None:
                raise ValueError("Either Z or A must be provided")
            A = Softmax.forward(Z)

        dZ = np.empty_like(A)
        for sample_index, probabilities in enumerate(A):
            jacobian = Softmax._jacobian(probabilities)
            dZ[sample_index] = dA[sample_index] @ jacobian

        return dZ

    @staticmethod
    def _jacobian(probabilities):
        """Return the softmax Jacobian for one sample."""
        diagonal_probabilities = np.diag(probabilities)
        probability_outer_product = np.outer(probabilities, probabilities)

        return diagonal_probabilities - probability_outer_product


class ReLU:
    """ReLU activation: g(z) = max(0, z)."""

    @staticmethod
    def forward(Z):
        """Compute ReLU activation."""
        return np.maximum(0, Z)

    @staticmethod
    def backward(dA, Z=None, A=None):
        """Return dZ, which is zero for inactive ReLU neurons."""
        if Z is None:
            if A is None:
                raise ValueError("Either Z or A must be provided")
            active_neurons = A > 0
        else:
            active_neurons = Z > 0

        dZ = np.array(dA, copy=True)
        dZ[~active_neurons] = 0

        return dZ


class DenseLayer:
    """
    Fully-connected layer storing a weight matrix W and bias vector b.

    The layer computes Z = inputs @ W + b, then applies its activation.
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
        """Initialize weights and biases once the input feature count is known."""
        self.input_features = input_features

        if rng is None:
            rng = np.random.default_rng()

        standard_deviation = self._weight_standard_deviation()
        self.W = rng.normal(0, standard_deviation, (input_features, self.units))
        self.b = np.zeros((1, self.units))

    def forward(self, inputs):
        """Compute and store this layer's linear and activation outputs."""
        self.inputs = np.asarray(inputs, dtype=float)
        self.Z = self.inputs @ self.W + self.b
        self.A = self._apply_activation(self.Z)

        return self.A

    def backward(self, gradient, output_layer=False):
        """
        Calculate parameter gradients and return the previous layer's gradient.

        ``gradient`` contains one gradient per sample. This method averages dW
        and db across the batch after calculating the per-sample derivatives.
        """
        dZ = self._calculate_dZ(gradient, output_layer)
        self._calculate_parameter_gradients(dZ)

        input_gradient = dZ @ self.W.T
        return input_gradient

    def _weight_standard_deviation(self):
        """Choose He initialization for ReLU and Xavier-style initialization otherwise."""
        if self.activation_name == "relu":
            return np.sqrt(2 / self.input_features)

        return np.sqrt(1 / self.input_features)

    def _apply_activation(self, Z):
        """Apply this layer's configured activation function."""
        if self.activation_name == "relu":
            return ReLU.forward(Z)
        if self.activation_name == "softmax":
            return Softmax.forward(Z)
        if self.activation_name == "sigmoid":
            return Sigmoid.forward(Z)
        if self.activation_name == "tanh":
            return Tanh.forward(Z)

        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def _calculate_dZ(self, gradient, output_layer):
        """Convert the incoming gradient into the linear-output gradient dZ."""
        gradient = np.asarray(gradient, dtype=float)

        if output_layer:
            # For a sigmoid/softmax output paired with cross-entropy, the MLP
            # supplies the fused per-sample gradient: prediction - target.
            return gradient

        if self.activation_name == "relu":
            return ReLU.backward(gradient, Z=self.Z, A=self.A)
        if self.activation_name == "softmax":
            return Softmax.backward(gradient, Z=self.Z, A=self.A)
        if self.activation_name == "sigmoid":
            return Sigmoid.backward(gradient, Z=self.Z, A=self.A)
        if self.activation_name == "tanh":
            return Tanh.backward(gradient, Z=self.Z, A=self.A)

        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def _calculate_parameter_gradients(self, dZ):
        """Calculate the average weight and bias gradients for this batch."""
        batch_size = self.inputs.shape[0]

        # Gradient for every input-to-neuron weight connection.
        transposed_inputs = self.inputs.T
        weight_gradient = transposed_inputs @ dZ
        bias_gradient = np.sum(dZ, axis=0, keepdims=True)

        self.dW = weight_gradient / batch_size
        self.db = bias_gradient / batch_size


class CategoricalCrossEntropy:
    """Categorical cross-entropy loss."""

    @staticmethod
    def compute(y_true, y_pred):
        """Compute the average categorical cross-entropy loss."""
        clipped_predictions = np.clip(y_pred, 1e-15, 1 - 1e-15)
        log_probabilities = np.log(clipped_predictions)
        total_loss = -np.sum(y_true * log_probabilities)
        number_of_samples = y_true.shape[0]

        return total_loss / number_of_samples

    @staticmethod
    def gradient(y_true, y_pred, eps=1e-15):
        """Return d(loss)/d(prediction), including the loss's sample average."""
        clipped_predictions = np.clip(y_pred, eps, 1 - eps)
        unaveraged_gradient = -(y_true / clipped_predictions)
        number_of_samples = y_true.shape[0]

        return unaveraged_gradient / number_of_samples


class BinaryCrossEntropy:
    """Binary cross-entropy: -mean(y * log(p) + (1 - y) * log(1 - p))."""

    @staticmethod
    def compute(y_true, y_pred):
        """Compute the average binary cross-entropy loss."""
        clipped_predictions = np.clip(y_pred, 1e-15, 1 - 1e-15)
        positive_loss = y_true * np.log(clipped_predictions)
        negative_loss = (1 - y_true) * np.log(1 - clipped_predictions)

        return -np.mean(positive_loss + negative_loss)

    @staticmethod
    def gradient(y_true, y_pred, eps=1e-15):
        """Return d(loss)/d(prediction), including the mean used by ``compute``."""
        clipped_predictions = np.clip(y_pred, eps, 1 - eps)
        positive_gradient = -(y_true / clipped_predictions)
        negative_gradient = (1 - y_true) / (1 - clipped_predictions)
        number_of_values = np.size(y_true)

        return (positive_gradient + negative_gradient) / number_of_values


class MultilayerPerceptron:
    """
    Coordinate DenseLayers, training, prediction, evaluation, and persistence.

    Training uses a sigmoid output with binary cross-entropy. The fused output
    gradient is averaged by ``DenseLayer.backward`` when it calculates dW/db.
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

    def fit(self, X, y, epochs, batch_size, learning_rate):
        """Train the neural network with mini-batch gradient descent."""
        self._validate_training_configuration(batch_size)
        X, y = self._prepare_training_data(X, y)
        self._reset_training_state()

        for _ in range(epochs):
            self._train_epoch(X, y, batch_size, learning_rate)
            self._record_epoch_results(X, y)

        return self

    def forward(self, X):
        """Propagate input X sequentially through every layer."""
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        return layer_output

    def backward(self, y_true, y_pred):
        """Backpropagate the binary cross-entropy error through all layers."""
        # The final sigmoid + BCE derivative simplifies to prediction - target.
        output_gradient = y_pred - y_true
        previous_layer_gradient = self.layers[-1].backward(
            output_gradient,
            output_layer=True,
        )

        for layer in reversed(self.layers[:-1]):
            previous_layer_gradient = layer.backward(previous_layer_gradient)

        return previous_layer_gradient

    def predict_proba(self, X):
        """Return predicted probabilities for input X."""
        X = np.asarray(X, dtype=float)
        return self.forward(X)

    def predict(self, X):
        """Return binary class predictions (0 or 1)."""
        probabilities = self.predict_proba(X)
        return self._probabilities_to_predictions(probabilities)

    def evaluate(self, X, y):
        """Compute binary loss, classification metrics, and a confusion matrix."""
        X = np.asarray(X, dtype=float)
        y_true = self._prepare_evaluation_labels(y)
        probabilities = self.predict_proba(X).reshape(-1)
        predictions = self._probabilities_to_predictions(probabilities)
        metrics = self._calculate_classification_metrics(y_true, predictions)

        return {
            "loss": BinaryCrossEntropy.compute(y_true, probabilities),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

    def save(self, filepath):
        """Save model topology, weights, biases, and scaler to JSON."""
        model_data = self._serialize()
        self._create_parent_folder(filepath)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(model_data, file, indent=4)

    @classmethod
    def load(cls, filepath):
        """Load model topology, weights, biases, and scaler from JSON."""
        model_data = cls._read_json(filepath)
        model = cls(seed=model_data.get("seed", 42))

        for saved_layer in model_data["layers"]:
            model._add_loaded_layer(saved_layer)

        model.scaler = model_data.get("scaler")
        return model

    def _validate_training_configuration(self, batch_size):
        """Check the training requirements that are independent of the data."""
        if len(self.layers) == 0:
            raise ValueError("Add at least one layer before training")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

    def _prepare_training_data(self, X, y):
        """Convert features and binary labels to the shapes used during training."""
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y)

        if len(features) != len(labels):
            raise ValueError("X and y must contain the same number of examples")

        labels = self._convert_text_labels(labels)
        labels = labels.astype(float).reshape(-1, 1)

        return features, labels

    def _reset_training_state(self):
        """Reset reproducible batch shuffling and the metric history."""
        np.random.seed(self.seed)
        self.history = {"loss": [], "accuracy": []}

    def _train_epoch(self, X, y, batch_size, learning_rate):
        """Shuffle the examples and train once on every mini-batch."""
        number_of_samples = len(X)
        shuffled_indexes = np.arange(number_of_samples)
        np.random.shuffle(shuffled_indexes)

        for start_index in range(0, number_of_samples, batch_size):
            end_index = start_index + batch_size
            batch_indexes = shuffled_indexes[start_index:end_index]

            batch_X = X[batch_indexes]
            batch_y = y[batch_indexes]
            self._train_batch(batch_X, batch_y, learning_rate)

    def _train_batch(self, batch_X, batch_y, learning_rate):
        """Run one forward pass, backward pass, and parameter update."""
        batch_predictions = self.forward(batch_X)
        self.backward(batch_y, batch_predictions)
        self._update_layers(learning_rate)

    def _update_layers(self, learning_rate):
        """Apply one gradient-descent update to each layer."""
        for layer in self.layers:
            weight_update = learning_rate * layer.dW
            bias_update = learning_rate * layer.db

            layer.W -= weight_update
            layer.b -= bias_update

    def _record_epoch_results(self, X, y):
        """Evaluate the current model and add its metrics to the history."""
        results = self.evaluate(X, y)
        self.history["loss"].append(results["loss"])
        self.history["accuracy"].append(results["accuracy"])

    @staticmethod
    def _convert_text_labels(labels):
        """Convert the dataset's M/B diagnosis labels into binary values."""
        if labels.dtype.kind in {"U", "S", "O"}:
            return (labels == "M").astype(float)

        return labels

    def _prepare_evaluation_labels(self, y):
        """Convert evaluation labels to a flat integer array."""
        labels = np.asarray(y)
        labels = self._convert_text_labels(labels)

        return labels.astype(int).reshape(-1)

    @staticmethod
    def _probabilities_to_predictions(probabilities):
        """Turn binary probabilities into 0/1 class predictions."""
        flat_probabilities = np.asarray(probabilities).reshape(-1)
        return (flat_probabilities >= 0.5).astype(int)

    @staticmethod
    def _calculate_classification_metrics(y_true, predictions):
        """Calculate accuracy, precision, recall, and a confusion matrix."""
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

        confusion_matrix = np.array([
            [true_negative, false_positive],
            [false_negative, true_positive],
        ])

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": confusion_matrix,
        }

    def _serialize(self):
        """Create the JSON-compatible representation used by ``save``."""
        layer_data = []
        for layer in self.layers:
            layer_data.append({
                "units": layer.units,
                "activation": layer.activation_name,
                "weights": layer.W.tolist(),
                "biases": layer.b.tolist(),
            })

        return {
            "seed": self.seed,
            "layers": layer_data,
            "scaler": getattr(self, "scaler", None),
        }

    @staticmethod
    def _create_parent_folder(filepath):
        """Create the model's parent folder when the path includes one."""
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def _read_json(filepath):
        """Read the JSON data used to restore a saved model."""
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    def _add_loaded_layer(self, saved_layer):
        """Recreate and append one layer from its saved JSON data."""
        weights = np.array(saved_layer["weights"], dtype=float)
        biases = np.array(saved_layer["biases"], dtype=float)

        layer = DenseLayer(saved_layer["units"], saved_layer["activation"])
        layer.build(weights.shape[0], rng=self.rng)
        layer.W = weights
        layer.b = biases

        self.add(layer)
