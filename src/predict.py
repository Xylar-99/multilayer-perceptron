"""Predict classes for feature-only or labeled CSV files."""

import pandas as pd

from .data import load_scaler, scale_features
from .model import BCE, MultilayerPerceptron



class Predictor:
    """Predict classes from a CSV with ID, optional diagnosis, and features."""

    def __init__(self, model_path: str, scaler_path: str, input_csv: str, output_csv: str):
        self.model = MultilayerPerceptron.load(model_path)
        self.scaler = load_scaler(scaler_path)
        self.input_csv = input_csv
        self.output_csv = output_csv


    def predict(self):
        """Load data, scale features, and save predictions."""

        data = pd.read_csv(self.input_csv, header=None)
        ids = data.iloc[:, 0]
        labels = None

        if data.shape[1] == self.model.input_features + 2:
            labels = data.iloc[:, 1].astype("string").str.strip()
            if not labels.isin(["B", "M"]).all():
                raise ValueError("Diagnosis column must contain B or M labels.")
            features = data.iloc[:, 2:]
        elif data.shape[1] == self.model.input_features + 1:
            features = data.iloc[:, 1:]
        else:
            raise ValueError(
                "Prediction CSV must contain an ID, optional B/M diagnosis, "
                f"and {self.model.input_features} numeric features."
            )

        try:
            features = features.to_numpy(dtype=float)
        except ValueError as error:
            raise ValueError("Prediction features must be numeric.") from error

        if features.shape[1] != self.model.input_features:
            raise ValueError(
                f"Expected {self.model.input_features} features, "
                f"found {features.shape[1]}"
            )

        if labels is None:
            features = scale_features(features, self.scaler)
        predictions = self.model.predict(features)

        if labels is not None:
            targets = (labels == "M").astype(int).to_numpy()
            loss = BCE.compute(targets, self.model.predict_proba(features))
            print(f"Binary cross-entropy: {loss:.4f}")

        output = pd.DataFrame({
            "id": ids,
            "predict": predictions,
        })

        output.to_csv(self.output_csv, index=False)
