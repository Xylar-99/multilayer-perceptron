"""Evaluate labeled data or predict classes for feature-only CSV files."""

import pandas as pd

from .data import Dataset, load_scaler, scale_features
from .model import MultilayerPerceptron





class Predictor:
    """Predict classes for feature-only CSV files or evaluate labeled data."""

    def __init__(self, model_path: str, scaler_path: str , input_csv: str, output_csv: str):
        self.model = MultilayerPerceptron.load(model_path)
        self.scaler = load_scaler(scaler_path)
        self.input_csv = input_csv
        self.output_csv = output_csv

    def predict(self):
        """Predict classes for feature-only CSV files."""


        data = pd.read_csv(self.input_csv)

        features = data.iloc[:, 1:].values.astype(float)
        scaled_features = scale_features(features, self.scaler)

        predictions = self.model.predict(scaled_features)

        predictions_df = pd.DataFrame(predictions, columns=["predict"])
        
        pd.DataFrame(predictions_df).to_csv(self.output_csv, index=False)