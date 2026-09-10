import os
import json
import numpy as np
import pandas as pd


class Dataset:
    """
    Encapsulates data arrays (features X, labels y).
    """

    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def cleanup(self):
        """Clean missing or invalid values."""
        data = pd.DataFrame(self.X)
        data["y"] = self.y
        data = data.dropna()
        self.X = data.drop(columns=["y"]).values
        self.y = data["y"].values
        return self

    def split(self, train_ratio):
        """Split this dataset into train and test datasets."""
        indices = np.random.permutation(len(self.X))
        train_size = int(train_ratio * len(self.X))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_X = self.X[train_indices]
        train_y = self.y[train_indices]
        test_X = self.X[test_indices]
        test_y = self.y[test_indices]

        return Dataset(train_X, train_y), Dataset(test_X, test_y)

    def fit_scaler(self):
        """
        Fit Min-Max scaler ONLY on this dataset (train set).
        Returns a dictionary with min and max for each feature.
        """
        X_arr = np.asarray(self.X, dtype=np.float64)
        scaler = {
            "type": "minmax",
            "min": np.min(X_arr, axis=0).tolist(),
            "max": np.max(X_arr, axis=0).tolist()
        }
        return scaler

    def scale(self, scaler):
        """
        Scale this dataset using an existing scaler:
        X_scaled = (X - min) / (max - min)
        """
        if scaler is None:
            return self

        X_arr = np.asarray(self.X, dtype=np.float64)
        min_val = np.array(scaler["min"], dtype=np.float64)
        max_val = np.array(scaler["max"], dtype=np.float64)

        diff = max_val - min_val
        # Prevent division by zero if max == min
        diff[diff == 0.0] = 1.0

        self.X = (X_arr - min_val) / diff
        return self

    def save_csv(self, filepath):
        """Save this dataset to CSV."""
        data = pd.DataFrame(self.X)
        data["y"] = self.y
        data.to_csv(filepath, index=False, header=False)

    @staticmethod
    def save_scaler(scaler, filepath):
        """Save scaler parameters to JSON file using standard json library."""

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scaler, f, indent=4)


    @staticmethod
    def load_scaler(filepath):
        """Load scaler parameters from JSON file using standard json library."""
        with open(filepath, "r", encoding="utf-8") as f:
            scaler = json.load(f)
        return scaler


    @staticmethod
    def load_csv(filepath):
        """Load CSV using pandas and return a Dataset object."""
        raw_data = pd.read_csv(filepath, header=None)
        diagnosis_col = None

        for col in raw_data.columns:
            values = set(raw_data[col].dropna().astype(str).str.strip())
            if values.issubset({"M", "B"}) and len(values) > 0:
                diagnosis_col = col
                break

        if diagnosis_col is None:
            raise ValueError("Diagnosis column not found")

        y = raw_data.iloc[:, diagnosis_col].values

        # Drop diagnosis column and column 0 (ID) if present
        cols_to_drop = [diagnosis_col]
        if 0 not in cols_to_drop and len(raw_data.columns) > 31:
            cols_to_drop.append(0)

        X = raw_data.drop(columns=cols_to_drop).values
        return Dataset(X, y)