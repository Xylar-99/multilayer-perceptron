"""Dataset loading, cleaning, scaling, and scaler persistence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_scaler(scaler, filepath):
    """Save Min-Max scaling parameters as JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(scaler, file, indent=4)


def load_scaler(filepath):
    """Load Min-Max scaling parameters from JSON."""
    with Path(filepath).open("r", encoding="utf-8") as file:
        return json.load(file)


def scale_features(features, scaler):
    """Return features scaled with a previously fitted Min-Max scaler."""
    features = np.asarray(features, dtype=float)
    feature_minimums = np.asarray(scaler["min"], dtype=float)
    feature_maximums = np.asarray(scaler["max"], dtype=float)

    feature_ranges = feature_maximums - feature_minimums
    feature_ranges[feature_ranges == 0.0] = 1.0
    return (features - feature_minimums) / feature_ranges


def features_need_scaling(features):
    """Recognise raw Wisconsin measurements rather than prepared 0-to-1 data."""
    return np.max(np.abs(features)) > 10


class Dataset:
    """Store feature values (``X``) and their diagnosis labels (``y``)."""

    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    @classmethod
    def from_csv(cls, filepath, allow_missing_diagnosis=False):
        """Load a headerless CSV and build a labeled dataset when possible."""
        raw_data = pd.read_csv(filepath, header=None)
        return cls.from_frame(raw_data, allow_missing_diagnosis)

    @classmethod
    def from_frame(cls, raw_data, allow_missing_diagnosis=False):
        """Build a dataset from a DataFrame containing an ``M``/``B`` column."""
        diagnosis_column = None
        for column in raw_data.columns:
            values = raw_data[column].dropna().astype(str).str.strip()
            if not values.empty and set(values).issubset({"M", "B"}):
                diagnosis_column = column
                break

        if diagnosis_column is None:
            if allow_missing_diagnosis:
                return None
            raise ValueError("Diagnosis column not found")

        columns_to_remove = [diagnosis_column]
        if diagnosis_column != 0 and raw_data.shape[1] > 31:
            columns_to_remove.append(0)

        features = raw_data.drop(columns=columns_to_remove).to_numpy()
        labels = raw_data.iloc[:, diagnosis_column].astype("string").str.strip().to_numpy()
        return cls(features, labels)

    def clean(self):
        """Remove rows with missing or non-numeric features and diagnoses."""
        feature_data = pd.DataFrame(self.X).apply(pd.to_numeric, errors="coerce")
        label_data = pd.Series(self.y, name="diagnosis")
        if pd.api.types.is_string_dtype(label_data):
            label_data = label_data.astype("string").str.strip()

        complete_rows = pd.concat([feature_data, label_data], axis=1).dropna()
        if complete_rows.empty:
            raise ValueError("No complete rows remain after cleaning the dataset")

        self.X = complete_rows.iloc[:, :-1].to_numpy(dtype=float)
        self.y = complete_rows.iloc[:, -1].to_numpy()
        return self

    def split(self, train_ratio, rng=None):
        """Randomly split this dataset into train and test datasets."""
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        random_generator = rng if rng is not None else np.random.default_rng()
        shuffled_indices = random_generator.permutation(len(self.X))
        train_size = int(train_ratio * len(self.X))

        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]
        return (
            Dataset(self.X[train_indices], self.y[train_indices]),
            Dataset(self.X[test_indices], self.y[test_indices]),
        )

    def fit_scaler(self):
        """Fit Min-Max scaling parameters using only this dataset's features."""
        features = np.asarray(self.X, dtype=float)
        if len(features) == 0:
            raise ValueError("Cannot fit a scaler to an empty dataset")

        return {
            "type": "minmax",
            "min": np.min(features, axis=0).tolist(),
            "max": np.max(features, axis=0).tolist(),
        }

    def scale(self, scaler):
        """Scale this dataset's features in place with an existing scaler."""
        if scaler is not None:
            self.X = scale_features(self.X, scaler)
        return self

    def save_csv(self, filepath):
        """Save features and labels to a headerless CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = pd.DataFrame(self.X)
        data["diagnosis"] = self.y
        data.to_csv(path, index=False, header=False)
