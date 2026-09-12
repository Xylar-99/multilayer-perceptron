import numpy as np
import pandas as pd

from .data import Dataset
from .model import MultilayerPerceptron


class ModelPredictor:
    """Load a saved model, then evaluate labeled data or predict new rows."""

    @staticmethod
    def run(dataset_path, model_path, scaler_path=None):
        """Run evaluation for labeled data or prediction for feature-only data.

        ``scaler_path`` is optional. When supplied, its scaler replaces the
        scaler saved inside the model, which is useful for older model files.
        """
        print(f"Dataset Path: {dataset_path}, Model Path: {model_path}")
        model = ModelPredictor._load_model(model_path, scaler_path)

        labeled_data = ModelPredictor._load_labeled_dataset(dataset_path)
        if labeled_data is not None:
            ModelPredictor._scale_dataset_if_needed(labeled_data, model)
            results = model.evaluate(labeled_data.X, labeled_data.y)
            ModelPredictor.print_report(results)
            return results

        features = ModelPredictor._load_feature_only_data(dataset_path, model)
        scaled_features = ModelPredictor._scale_features_if_needed(features, model)
        probabilities = model.predict_proba(scaled_features).reshape(-1)
        predictions = model.predict(scaled_features)

        ModelPredictor._print_predictions(probabilities, predictions)
        return {
            "probabilities": probabilities,
            "predictions": predictions,
        }

    @staticmethod
    def print_report(results, title="Test Set Evaluation"):
        """Format and print evaluation metrics."""
        print(f"\n{title}")
        print(f"Loss:      {results['loss']:.4f}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print("Confusion matrix:")
        print(results["confusion_matrix"])

    @staticmethod
    def _load_model(model_path, scaler_path):
        """Load the model and optionally replace its saved scaler."""
        model = MultilayerPerceptron.load(model_path)
        if scaler_path is not None:
            model.scaler = Dataset.load_scaler(scaler_path)
        return model

    @staticmethod
    def _load_labeled_dataset(dataset_path):
        """Return cleaned labeled data, or ``None`` for a feature-only file."""
        try:
            return Dataset.load_csv(dataset_path).cleanup()
        except ValueError as error:
            if str(error) == "Diagnosis column not found":
                return None
            raise

    @staticmethod
    def _scale_dataset_if_needed(dataset, model):
        """Scale labeled raw data, while leaving already scaled data alone."""
        if model.scaler is not None and ModelPredictor._features_need_scaling(dataset.X):
            dataset.scale(model.scaler)
        return dataset

    @staticmethod
    def _load_feature_only_data(dataset_path, model):
        """Read numeric features and allow one leading identifier column."""
        raw_data = pd.read_csv(dataset_path, header=None)
        expected_feature_count = model.layers[0].input_features
        feature_data = ModelPredictor._remove_identifier_column(
            raw_data,
            expected_feature_count,
        )

        ModelPredictor._validate_feature_count(
            feature_data,
            expected_feature_count,
        )
        return feature_data.astype(float).values

    @staticmethod
    def _remove_identifier_column(raw_data, expected_feature_count):
        """Drop a leading ID column when the file has exactly one extra column."""
        if raw_data.shape[1] == expected_feature_count + 1:
            return raw_data.iloc[:, 1:]
        return raw_data

    @staticmethod
    def _validate_feature_count(feature_data, expected_feature_count):
        """Raise a clear error when the CSV cannot be fed to the model."""
        actual_feature_count = feature_data.shape[1]
        if actual_feature_count != expected_feature_count:
            raise ValueError(
                f"Expected {expected_feature_count} feature columns, "
                f"but found {actual_feature_count}"
            )

    @staticmethod
    def _scale_features_if_needed(features, model):
        """Apply the model's scaler to raw feature values when available."""
        if model.scaler is None or not ModelPredictor._features_need_scaling(features):
            return features

        feature_dataset = Dataset(X=features)
        feature_dataset.scale(model.scaler)
        return feature_dataset.X

    @staticmethod
    def _features_need_scaling(features):
        """Identify raw breast-cancer measurements by their larger magnitudes."""
        return np.max(np.abs(features)) > 10

    @staticmethod
    def _print_predictions(probabilities, predictions):
        """Print one class and probability for each feature-only row."""
        print("\nPredictions")
        for row_number, (probability, prediction) in enumerate(
            zip(probabilities, predictions),
            start=1,
        ):
            print(
                f"Row {row_number}: "
                f"class={prediction}, probability={probability:.4f}"
            )
