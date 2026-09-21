"""Predict classes for feature-only or labeled CSV files."""

import pandas as pd

from .data import load_scaler, scale_features
from .model import BCE, MultilayerPerceptron

import numpy as np

class Predictor:
    """Predict classes from a CSV with ID, optional diagnosis, and features."""

    def __init__(self, model_path: str, scaler_path: str, input_csv: str, output_csv: str):
        self.model = MultilayerPerceptron.load(model_path)
        self.scaler = load_scaler(scaler_path)
        self.input_csv = input_csv
        self.output_csv = output_csv


    def predict(self):
        """Load labeled data, predict classes, evaluate with BCE, and save predictions."""

        data = pd.read_csv(self.input_csv, header=None)

        expected_columns = self.model.input_features + 2

        if data.shape[1] != expected_columns:
            raise ValueError(
                "Prediction CSV must contain an ID, a B/M diagnosis, "
                f"and {self.model.input_features} numeric features."
            )

        ids = data.iloc[:, 0]
        labels = data.iloc[:, 1].astype("string").str.strip()

        if not labels.isin(["B", "M"]).all():
            raise ValueError("Diagnosis column must contain B or M labels.")

        features = data.iloc[:, 2:]

        features = features.to_numpy(dtype=float)
        features = scale_features(features, self.scaler)

        predictions = self.model.predict(features)

        targets = (labels == "M").astype(int).to_numpy()
        probabilities = self.model.predict_proba(features)

        # Compute binary cross-entropy loss
        loss = BCE.compute(targets, probabilities)
        print(f"Binary cross-entropy: {loss:.4f}")

        # Compute accuracy
        accuracy = np.mean(predictions == labels.to_numpy()) * 100
        print(f"Accuracy: {accuracy:.2f}%")

        output = pd.DataFrame({
            "id": ids,
            "predict": predictions,
        })

        output.to_csv(self.output_csv, index=False)
