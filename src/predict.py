"""Evaluate labeled data or predict classes for feature-only CSV files."""

import pandas as pd

from .data import Dataset, features_need_scaling, load_scaler, scale_features
from .model import MultilayerPerceptron


def print_evaluation(results, title="Test Set Evaluation"):
    """Print the evaluation dictionary returned by ``MultilayerPerceptron``."""
    print(f"\n{title}")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print("Confusion matrix:")
    print(results["confusion_matrix"])


def scale_if_needed(features, scaler):
    """Scale raw breast-cancer measurements, but leave prepared CSVs unchanged."""
    if scaler is not None and features_need_scaling(features):
        return scale_features(features, scaler)
    return features


def predict_from_file(dataset_path, model_path, scaler_path=None):
    """Evaluate labeled data or predict a feature-only CSV with a saved model."""
    model = MultilayerPerceptron.load(model_path)
    if scaler_path is not None:
        model.scaler = load_scaler(scaler_path)

    raw_data = pd.read_csv(dataset_path, header=None)
    if raw_data.empty:
        raise ValueError("Prediction CSV contains no rows")
    labeled_data = Dataset.from_frame(raw_data, allow_missing_diagnosis=True)
    if labeled_data is not None:
        labeled_data.clean()
        labeled_data.X = scale_if_needed(labeled_data.X, model.scaler)
        results = model.evaluate(labeled_data.X, labeled_data.y)
        print_evaluation(results)
        return results

    expected_feature_count = model.layers[0].input_features
    if raw_data.shape[1] == expected_feature_count + 1:
        raw_data = raw_data.iloc[:, 1:]
    if raw_data.shape[1] != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} feature columns, but found {raw_data.shape[1]}"
        )

    features = raw_data.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    features = scale_if_needed(features, model.scaler)
    probabilities = model.predict_proba(features).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)

    print("\nPredictions")
    for row_number, (probability, prediction) in enumerate(zip(probabilities, predictions), start=1):
        print(f"Row {row_number}: class={prediction}, probability={probability:.4f}")
    return {"probabilities": probabilities, "predictions": predictions}
