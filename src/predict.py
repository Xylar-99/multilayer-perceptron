"""Predict classes for feature-only CSV files."""

import pandas as pd

from .data import load_scaler, scale_features
from .model import MultilayerPerceptron



class Predictor:
    """Predict classes from a CSV with ID + features."""

    def __init__(self, model_path: str, scaler_path: str, input_csv: str, output_csv: str):
        self.model = MultilayerPerceptron.load(model_path)
        self.scaler = load_scaler(scaler_path)
        self.input_csv = input_csv
        self.output_csv = output_csv


    def predict(self):
        """Load data, scale features, and save predictions."""

        data = pd.read_csv(self.input_csv, header=None)

        ids = data.iloc[:, 0]
        features = data.iloc[:, 1:].to_numpy(dtype=float)

        if features.shape[1] != self.model.input_features:
            raise ValueError(
                f"Expected {self.model.input_features} features, "
                f"found {features.shape[1]}"
            )

        features = scale_features(features, self.scaler)
        predictions = self.model.predict(features)

        output = pd.DataFrame({
            "id": ids,
            "predict": predictions,
        })

        output.to_csv(self.output_csv, index=False)
