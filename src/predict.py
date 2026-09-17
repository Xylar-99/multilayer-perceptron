"""Evaluate labeled data or predict classes for feature-only CSV files."""

import pandas as pd

from .data import Dataset, load_scaler, scale_features
from .model import MultilayerPerceptron





class Predictor:
    """Predict classes for feature-only CSV files or evaluate labeled data."""

    def __init__(self, model_path: str, scaler_path: str):
        self.model = MultilayerPerceptron.load(model_path)
        self.scaler = load_scaler(scaler_path)

    def predict(self, input_csv: str , output_csv: str ):
        """Predict classes for feature-only CSV files."""


        data = pd.read_csv(input_csv, header=None)
        features = data.iloc[:, 1:].to_numpy()

        scaled_features = scale_features(features, self.scaler)
        predictions = self.model.predict(scaled_features)

        self.save_predictions(predictions, output_csv)



    def save_predictions(self, predictions, output_csv: str):
        """Save predictions to a CSV file."""

        index = range(1, len(predictions) + 1)
        labels = ["B" if pred == 0 else "M" for pred in predictions]

        pd.DataFrame(labels).to_csv(output_csv, index=False, header=False)
        print(f"Predictions saved to {output_csv}")

    