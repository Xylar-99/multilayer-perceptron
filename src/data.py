"""Dataset loading, cleaning, and scaling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def scale_features(features, scaler):
    """Scale features using Min-Max values."""
    minimum = np.asarray(scaler["min"])
    maximum = np.asarray(scaler["max"])

    ranges = maximum - minimum
    ranges[ranges == 0] = 1

    return (features - minimum) / ranges


def save_scaler(scaler, filepath):
    """Save Min-Max values to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(scaler, f)


def load_scaler(filepath):
    """Load Min-Max values from a JSON file."""
    with open(filepath, "r") as f:
        scaler = json.load(f)

    return scaler


class Dataset:
    """Store feature values (X) and labels (y)."""

    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    @classmethod
    def from_csv(cls, filepath):
        data = pd.read_csv(filepath, header=None)
        
        labels = data.iloc[:, 1].to_numpy()
        features = data.iloc[:, 2:].to_numpy()

        return cls(features, labels)

    def clean(self):
        """Convert features to numbers and remove incomplete rows."""
        self.X = pd.DataFrame(self.X).apply(
            pd.to_numeric, errors="coerce"
        )
        self.y = pd.Series(self.y).astype("string").str.strip()

        valid_rows = self.X.notna().all(axis=1) & self.y.notna()

        self.X = self.X[valid_rows].to_numpy(dtype=float)
        self.y = self.y[valid_rows].to_numpy()

        if len(self.X) == 0:
            raise ValueError("No valid rows remain after cleaning")

        return self

    def split(self, train_ratio):
        """Randomly split the dataset into train and test sets."""
        indices = np.random.permutation(len(self.X))
        train_size = int(train_ratio * len(self.X))

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train = Dataset(self.X[train_indices], self.y[train_indices])
        test = Dataset(self.X[test_indices], self.y[test_indices])

        return train, test

    def fit_scaler(self):
        """Get Min-Max values from the features."""
        return {
            "min": np.min(self.X, axis=0).tolist(),
            "max": np.max(self.X, axis=0).tolist(),
        }

    def save_csv(self, filepath):
        """Save dataset to a headerless CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = pd.DataFrame(self.X)
        data.insert(0, "id", range(1, len(data) + 1))
        data.insert(1, "diagnosis", self.y)
        data.to_csv(path, index=False, header=False)

    def scale(self, scaler):
        """Scale features using Min-Max values."""
        self.X = scale_features(self.X, scaler)
