import json
from pathlib import Path

import numpy as np
import pandas as pd


class Dataset:
    """Store feature values (``X``) and their labels (``y``)."""

    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    @staticmethod
    def load_csv(filepath):
        """Load a labeled CSV file and return its features and diagnoses."""
        raw_data = pd.read_csv(filepath, header=None)
        diagnosis_column = Dataset._find_diagnosis_column(raw_data)

        if diagnosis_column is None:
            raise ValueError("Diagnosis column not found")

        features = Dataset._extract_features(raw_data, diagnosis_column)
        labels = raw_data.iloc[:, diagnosis_column].to_numpy()
        return Dataset(features, labels)

    def cleanup(self):
        """Remove rows with a missing feature value or diagnosis."""
        combined_data = self._combine_features_and_labels()
        complete_rows = combined_data.dropna()

        self.X = complete_rows.iloc[:, :-1].to_numpy()
        self.y = complete_rows.iloc[:, -1].to_numpy()
        return self

    def split(self, train_ratio, rng=None):
        """Randomly split this dataset into training and test datasets.

        Pass a NumPy random generator through ``rng`` when a repeatable split
        is needed. Without one, this keeps using NumPy's normal global random
        state, as before.
        """
        number_of_samples = len(self.X)
        random_generator = rng if rng is not None else np.random
        shuffled_indices = random_generator.permutation(number_of_samples)

        train_size = int(train_ratio * number_of_samples)
        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]

        train_dataset = Dataset(self.X[train_indices], self.y[train_indices])
        test_dataset = Dataset(self.X[test_indices], self.y[test_indices])
        return train_dataset, test_dataset

    def fit_scaler(self):
        """Fit a Min-Max scaler using only this dataset's feature values."""
        features = np.asarray(self.X, dtype=np.float64)
        feature_minimums = np.min(features, axis=0)
        feature_maximums = np.max(features, axis=0)

        return {
            "type": "minmax",
            "min": feature_minimums.tolist(),
            "max": feature_maximums.tolist(),
        }

    def scale(self, scaler):
        """Scale features with an already-fitted Min-Max scaler."""
        if scaler is None:
            return self

        features = np.asarray(self.X, dtype=np.float64)
        feature_minimums = np.asarray(scaler["min"], dtype=np.float64)
        feature_maximums = np.asarray(scaler["max"], dtype=np.float64)

        feature_ranges = feature_maximums - feature_minimums
        # A constant feature has no range, so divide it by one instead of zero.
        feature_ranges[feature_ranges == 0.0] = 1.0

        self.X = (features - feature_minimums) / feature_ranges
        return self

    def save_csv(self, filepath):
        """Save features and labels to a headerless CSV file."""
        self._create_parent_directory(filepath)

        combined_data = self._combine_features_and_labels()
        combined_data.to_csv(filepath, index=False, header=False)

    @staticmethod
    def save_scaler(scaler, filepath):
        """Save scaler parameters in a JSON file."""
        Dataset._create_parent_directory(filepath)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(scaler, file, indent=4)

    @staticmethod
    def load_scaler(filepath):
        """Load scaler parameters from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _find_diagnosis_column(raw_data):
        """Return the column that contains only ``M`` and ``B`` labels."""
        for column in raw_data.columns:
            non_missing_values = raw_data[column].dropna()
            diagnosis_values = set(
                non_missing_values.astype(str).str.strip()
            )

            if diagnosis_values and diagnosis_values.issubset({"M", "B"}):
                return column

        return None

    @staticmethod
    def _extract_features(raw_data, diagnosis_column):
        """Remove the diagnosis and the optional identifier column."""
        columns_to_remove = [diagnosis_column]
        has_identifier_column = (
            diagnosis_column != 0 and len(raw_data.columns) > 31
        )

        if has_identifier_column:
            columns_to_remove.append(0)

        return raw_data.drop(columns=columns_to_remove).to_numpy()

    def _combine_features_and_labels(self):
        """Build one table so rows can be cleaned or saved together."""
        feature_data = pd.DataFrame(self.X)
        label_data = pd.Series(self.y)
        return pd.concat([feature_data, label_data], axis=1)

    @staticmethod
    def _create_parent_directory(filepath):
        """Create the destination directory when it does not exist yet."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
