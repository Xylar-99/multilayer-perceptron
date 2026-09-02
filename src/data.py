import csv
import numpy as np


class StandardScaler:
    """
    Standardizes features by computing mean and standard deviation:
      z = (x - mean) / std
    Fitted strictly on training data to prevent data leakage.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True)
        self.std[self.std == 0.0] = 1.0  # Prevent divide-by-zero
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transforming.")
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self):
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None
        }

    def from_dict(self, data):
        if data and data.get('mean') is not None and data.get('std') is not None:
            self.mean = np.array(data['mean'], dtype=np.float64)
            self.std = np.array(data['std'], dtype=np.float64)
        return self


class Dataset:
    """
    Dataset wrapper holding feature matrix X and label vector y.
    Handles CSV loading, train/validation splitting, and mini-batch generation.
    """
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    @property
    def num_features(self):
        return self.X.shape[1]

    @classmethod
    def from_csv(cls, filepath):
        """Loads Wisconsin Breast Cancer CSV dataset."""
        rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                if not line or len(line) < 2:
                    continue
                rows.append(line)

        if not rows:
            raise ValueError(f"No data found in '{filepath}'.")

        # Skip header if first row has text labels
        if rows[0][1].strip().upper() not in ('M', 'B', '0', '1'):
            rows = rows[1:]

        labels = []
        features = []
        for r in rows:
            label = 1 if r[1].strip().upper() in ('M', '1') else 0
            feats = [float(val) for val in r[2:]]
            labels.append(label)
            features.append(feats)

        return cls(X=np.array(features, dtype=np.float64), y=np.array(labels, dtype=np.int64))

    def split(self, train_ratio=0.8, seed=42):
        """Splits into (train_dataset, val_dataset) deterministically."""
        rng = np.random.RandomState(seed)
        indices = np.arange(len(self))
        rng.shuffle(indices)

        split_idx = int(len(self) * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        return Dataset(self.X[train_idx], self.y[train_idx]), Dataset(self.X[val_idx], self.y[val_idx])

    def save_csv(self, filepath):
        """Exports dataset to raw CSV format (ID, Diagnosis, Features)."""
        import os
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for i in range(len(self.X)):
                diag = 'M' if self.y[i] == 1 else 'B'
                feats = ','.join(str(v) for v in self.X[i])
                f.write(f"{i},{diag},{feats}\n")

    def to_one_hot(self, num_classes=2):
        """Converts labels y into one-hot matrix."""
        N = len(self.y)
        one_hot = np.zeros((N, num_classes), dtype=np.float64)
        one_hot[np.arange(N), self.y] = 1.0
        return one_hot

    def batches(self, batch_size=8, shuffle=True, rng=None):
        """Yields (X_batch, y_batch) minibatches."""
        N = len(self.X)
        indices = np.arange(N)
        if shuffle:
            if rng is None:
                rng = np.random
            rng.shuffle(indices)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

