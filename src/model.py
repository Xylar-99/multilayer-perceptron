import os
import json
import numpy as np
from src.data import StandardScaler, Dataset


# ==========================================
# 1. Activation Functions
# ==========================================
class Sigmoid:
    @staticmethod
    def forward(Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

    @staticmethod
    def backward(dA, Z=None, A=None):
        if A is None:
            A = Sigmoid.forward(Z)
        return dA * A * (1.0 - A)


class ReLU:
    @staticmethod
    def forward(Z):
        return np.maximum(0.0, Z)

    @staticmethod
    def backward(dA, Z=None, A=None):
        val = Z if Z is not None else A
        return dA * (val > 0.0).astype(np.float64)


class Tanh:
    @staticmethod
    def forward(Z):
        return np.tanh(Z)

    @staticmethod
    def backward(dA, Z=None, A=None):
        if A is None:
            A = Tanh.forward(Z)
        return dA * (1.0 - A ** 2)


class Softmax:
    @staticmethod
    def forward(Z):
        shifted = Z - np.max(Z, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def backward(dA, Z=None, A=None):
        if A is None:
            A = Softmax.forward(Z)
        dot = np.sum(dA * A, axis=-1, keepdims=True)
        return A * (dA - dot)


def get_activation(name):
    name = str(name).lower().strip().replace('_', '').replace('-', '')
    if name == 'sigmoid':
        return Sigmoid
    elif name == 'relu':
        return ReLU
    elif name == 'tanh':
        return Tanh
    elif name == 'softmax':
        return Softmax
    raise ValueError(f"Unknown activation '{name}'. Choose: sigmoid, relu, tanh, softmax.")


# ==========================================
# 2. Dense Layer
# ==========================================
class Dense:
    """Fully-connected layer holding weights W and biases b."""
    def __init__(self, units, activation='sigmoid', initializer='heUniform', input_dim=None):
        self.units = int(units)
        self.activation_name = activation
        self.activation = get_activation(activation) if activation else None
        self.initializer = initializer
        self.input_dim = input_dim

        self.W = None
        self.b = None
        self.dW = None
        self.db = None

        self.inputs = None
        self.Z = None
        self.A = None

        if input_dim is not None:
            self.build(input_dim)

    def build(self, input_dim, rng=None):
        if rng is None:
            rng = np.random
        self.input_dim = input_dim
        init = str(self.initializer).lower().replace('_', '')

        if 'he' in init:
            limit = np.sqrt(6.0 / input_dim)
            self.W = rng.uniform(-limit, limit, (input_dim, self.units))
        elif 'xavier' in init or 'glorot' in init:
            limit = np.sqrt(6.0 / (input_dim + self.units))
            self.W = rng.uniform(-limit, limit, (input_dim, self.units))
        else:
            self.W = rng.normal(0.0, 0.05, (input_dim, self.units))

        self.b = np.zeros((1, self.units), dtype=np.float64)

    def forward(self, inputs):
        self.inputs = inputs
        self.Z = np.dot(inputs, self.W) + self.b
        self.A = self.activation.forward(self.Z) if self.activation else self.Z
        return self.A

    def backward(self, grad, is_delta=False):
        m = self.inputs.shape[0]
        dZ = grad if is_delta else (self.activation.backward(grad, Z=self.Z, A=self.A) if self.activation else grad)

        self.dW = np.dot(self.inputs.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        return np.dot(dZ, self.W.T)


# Alias
DenseLayer = Dense


# ==========================================
# 3. Loss Functions
# ==========================================
class BinaryCrossEntropy:
    @staticmethod
    def compute(y_true, y_pred, eps=1e-15):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        p = y_pred[:, 1] if (y_pred.ndim == 2 and y_pred.shape[1] == 2) else y_pred.squeeze()
        y = y_true[:, 1] if (y_true.ndim == 2 and y_true.shape[1] == 2) else y_true.squeeze()
        p = np.clip(p, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class CategoricalCrossEntropy:
    @staticmethod
    def compute(y_true, y_pred, eps=1e-15):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.ndim == 1:
            y_true = np.column_stack([1.0 - y_true, y_true])
        if y_pred.ndim == 1:
            y_pred = np.column_stack([1.0 - y_pred, y_pred])
        p = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=-1)))


def get_loss(name):
    name = str(name).lower().replace('_', '')
    if 'binary' in name:
        return BinaryCrossEntropy()
    return CategoricalCrossEntropy()


# ==========================================
# 4. Optimizers
# ==========================================
class SGD:
    def __init__(self, learning_rate=0.0314):
        self.lr = float(learning_rate)

    def update(self, layer, layer_idx):
        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(learning_rate)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m_W, self.m_b, self.v_W, self.v_b = {}, {}, {}, {}
        self.t = 0

    def update(self, layer, layer_idx):
        if layer_idx not in self.m_W:
            self.m_W[layer_idx] = np.zeros_like(layer.W)
            self.m_b[layer_idx] = np.zeros_like(layer.b)
            self.v_W[layer_idx] = np.zeros_like(layer.W)
            self.v_b[layer_idx] = np.zeros_like(layer.b)

        self.t += 1
        self.m_W[layer_idx] = self.beta1 * self.m_W[layer_idx] + (1.0 - self.beta1) * layer.dW
        self.m_b[layer_idx] = self.beta1 * self.m_b[layer_idx] + (1.0 - self.beta1) * layer.db
        self.v_W[layer_idx] = self.beta2 * self.v_W[layer_idx] + (1.0 - self.beta2) * (layer.dW ** 2)
        self.v_b[layer_idx] = self.beta2 * self.v_b[layer_idx] + (1.0 - self.beta2) * (layer.db ** 2)

        m_W_hat = self.m_W[layer_idx] / (1.0 - self.beta1 ** self.t)
        m_b_hat = self.m_b[layer_idx] / (1.0 - self.beta1 ** self.t)
        v_W_hat = self.v_W[layer_idx] / (1.0 - self.beta2 ** self.t)
        v_b_hat = self.v_b[layer_idx] / (1.0 - self.beta2 ** self.t)

        layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


def get_optimizer(name, learning_rate=0.0314):
    name = str(name).lower().replace('_', '')
    if 'adam' in name:
        return Adam(learning_rate=learning_rate)
    return SGD(learning_rate=learning_rate)


# ==========================================
# 5. Multilayer Perceptron Model
# ==========================================
class MultilayerPerceptron:
    """Chains Dense layers, manages forward/backward passes, training, and JSON persistence."""
    def __init__(self, layers=None, seed=42):
        self.layers = []
        self.scaler = StandardScaler()
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        if layers:
            for l in layers:
                self.add(l)

    def add(self, layer):
        self.layers.append(layer)

    @classmethod
    def create_network(cls, layers_list, seed=42):
        mlp = cls(seed=seed)
        for layer in layers_list:
            mlp.add(layer)
        return mlp

    def _build(self, input_dim):
        curr_dim = input_dim
        for layer in self.layers:
            if layer.W is None:
                layer.build(curr_dim, rng=self.rng)
            curr_dim = layer.units

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_true, y_pred):
        delta = y_pred - y_true
        dA = self.layers[-1].backward(delta, is_delta=True)
        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA, is_delta=False)

    def fit(self, train_data, val_data=None, epochs=84, batch_size=8,
            learning_rate=0.0314, optimizer='sgd', loss='categoricalCrossentropy',
            early_stopping=None, verbose=True):

        X_train, y_train = (train_data.X, train_data.y) if isinstance(train_data, Dataset) else train_data
        X_val, y_val = (val_data.X, val_data.y) if isinstance(val_data, Dataset) else (val_data or (None, None))

        # Fit scaler on training data only
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val) if X_val is not None else None

        N, D = X_train_norm.shape
        self._build(D)

        # Labels format
        num_classes = self.layers[-1].units
        if num_classes == 2:
            y_train_enc = Dataset(X_train, y_train).to_one_hot(2) if y_train.ndim == 1 else y_train
            y_val_enc = Dataset(X_val, y_val).to_one_hot(2) if (X_val is not None and y_val.ndim == 1) else y_val
        else:
            y_train_enc = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
            y_val_enc = y_val.reshape(-1, 1) if (X_val is not None and y_val.ndim == 1) else y_val

        loss_fn = get_loss(loss)
        opt = get_optimizer(optimizer, learning_rate=learning_rate) if isinstance(optimizer, str) else optimizer

        if verbose:
            print(f"x_train shape : {X_train.shape}")
            if X_val is not None:
                print(f"x_valid shape : {X_val.shape}")
            print(f"Training MLP: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}, optimizer={opt.__class__.__name__}\n")

        best_val_loss = float('inf')
        patience_count = 0
        num_batches = int(np.ceil(N / batch_size))

        for epoch in range(1, epochs + 1):
            indices = np.arange(N)
            self.rng.shuffle(indices)
            X_shuffled = X_train_norm[indices]
            y_shuffled = y_train_enc[indices]

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, N)

                X_b = X_shuffled[start:end]
                y_b = y_shuffled[start:end]

                y_pred_b = self.forward(X_b)
                self.backward(y_b, y_pred_b)

                for i, layer in enumerate(self.layers):
                    opt.update(layer, i)

            # Metrics
            y_train_pred = self.forward(X_train_norm)
            train_loss = loss_fn.compute(y_train_enc, y_train_pred)
            train_acc = float(np.mean(np.argmax(y_train_enc, axis=1) == np.argmax(y_train_pred, axis=1)))

            self.history['loss'].append(train_loss)
            self.history['acc'].append(train_acc)

            if X_val_norm is not None:
                y_val_pred = self.forward(X_val_norm)
                val_loss = loss_fn.compute(y_val_enc, y_val_pred)
                val_acc = float(np.mean(np.argmax(y_val_enc, axis=1) == np.argmax(y_val_pred, axis=1)))

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                if verbose:
                    print(f"epoch {epoch:02d}/{epochs:02d} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

                if early_stopping and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                elif early_stopping:
                    patience_count += 1
                    if patience_count >= early_stopping:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            else:
                if verbose:
                    print(f"epoch {epoch:02d}/{epochs:02d} - loss: {train_loss:.4f} - acc: {train_acc:.4f}")

        return self.history

    def predict_proba(self, X):
        X_norm = self.scaler.transform(X)
        return self.forward(X_norm)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) if (probs.ndim == 2 and probs.shape[1] == 2) else (probs.squeeze() >= 0.5).astype(int)

    def evaluate(self, X, y):
        probs = self.predict_proba(X)
        bce = BinaryCrossEntropy.compute(y, probs)
        preds = np.argmax(probs, axis=1) if (probs.ndim == 2 and probs.shape[1] == 2) else (probs.squeeze() >= 0.5).astype(int)
        y_true = y if y.ndim == 1 else np.argmax(y, axis=1)
        acc = float(np.mean(y_true == preds))

        TN = int(np.sum((y_true == 0) & (preds == 0)))
        FP = int(np.sum((y_true == 0) & (preds == 1)))
        FN = int(np.sum((y_true == 1) & (preds == 0)))
        TP = int(np.sum((y_true == 1) & (preds == 1)))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'bce_loss': bce, 'accuracy': acc, 'precision': float(precision),
            'recall': float(recall), 'f1_score': float(f1),
            'confusion_matrix': np.array([[TN, FP], [FN, TP]]), 'probabilities': probs
        }

    def save(self, filepath):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        data = {
            'scaler': self.scaler.to_dict(),
            'seed': self.seed,
            'layers': [
                {
                    'units': l.units, 'activation': l.activation_name,
                    'initializer': l.initializer, 'input_dim': l.input_dim,
                    'W': l.W.tolist() if l.W is not None else None,
                    'b': l.b.tolist() if l.b is not None else None
                }
                for l in self.layers
            ]
        }

        if filepath.endswith('.npy'):
            np.save(filepath, data, allow_pickle=True)
        else:
            if not filepath.endswith('.json'):
                filepath += '.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        print(f"> saving model '{filepath}' to disk...")

    @classmethod
    def load(cls, filepath):
        candidates = [
            filepath, os.path.join("output", filepath),
            filepath + ".json", os.path.join("output", filepath + ".json"),
            filepath + ".npy", os.path.join("output", filepath + ".npy"),
        ]
        actual = next((c for c in candidates if os.path.exists(c)), None)
        if not actual:
            raise FileNotFoundError(f"Model '{filepath}' not found.")

        if actual.endswith('.json'):
            with open(actual, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raw = np.load(actual, allow_pickle=True)
            data = raw.item() if isinstance(raw, np.ndarray) else raw

        mlp = cls(seed=data.get('seed', 42))
        mlp.scaler.from_dict(data.get('scaler', {}))

        for l_data in data['layers']:
            layer = Dense(
                units=l_data['units'], activation=l_data['activation'],
                initializer=l_data['initializer'], input_dim=l_data['input_dim']
            )
            layer.W = np.array(l_data['W'], dtype=np.float64)
            layer.b = np.array(l_data['b'], dtype=np.float64)
            mlp.add(layer)

        return mlp
