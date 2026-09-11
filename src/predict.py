import numpy as np
import pandas as pd

from .data import Dataset
from .model import MultilayerPerceptron


class ModelPredictor:
    """Handles loading saved model, running predictions, and evaluation."""
    @staticmethod
    def run(dataset_path, model_path):
        """Load model, load dataset, evaluate BCE and accuracy, and print report."""
        print(f"Dataset Path: {dataset_path}, Model Path: {model_path}")
        model = MultilayerPerceptron.load(model_path)

        try:
            # A labeled file is used for evaluation.
            dataset = Dataset.load_csv(dataset_path).cleanup()

            # The original CSV contains large feature values. The split CSV is
            # already scaled, so only scale clearly raw data here.
            if model.scaler is not None and np.max(np.abs(dataset.X)) > 10:
                dataset.scale(model.scaler)

            results = model.evaluate(dataset.X, dataset.y)
            ModelPredictor.print_report(results)
            return results
        except ValueError as error:
            if str(error) != "Diagnosis column not found":
                raise

        # A file without labels is used for normal prediction.
        raw_data = pd.read_csv(dataset_path, header=None)
        feature_count = model.layers[0].input_features

        if raw_data.shape[1] == feature_count + 1:
            # Support a feature file with an ID in the first column.
            raw_data = raw_data.iloc[:, 1:]

        if raw_data.shape[1] != feature_count:
            raise ValueError(
                f"Expected {feature_count} feature columns, "
                f"but found {raw_data.shape[1]}"
            )

        X = raw_data.astype(float).values
        if model.scaler is not None and np.max(np.abs(X)) > 10:
            minimum = np.array(model.scaler["min"])
            maximum = np.array(model.scaler["max"])
            difference = maximum - minimum
            difference[difference == 0] = 1
            X = (X - minimum) / difference

        probabilities = model.predict_proba(X).reshape(-1)
        predictions = model.predict(X)

        print("\nPredictions")
        for row_number, (probability, prediction) in enumerate(
            zip(probabilities, predictions),
            start=1,
        ):
            print(
                f"Row {row_number}: "
                f"class={prediction}, probability={probability:.4f}"
            )

        return {
            "probabilities": probabilities,
            "predictions": predictions,
        }

    @staticmethod
    def print_report(results, title="Test Set Evaluation"):
        """Format and print evaluation metrics table."""
        print(f"\n{title}")
        print(f"Loss:      {results['loss']:.4f}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print("Confusion matrix:")
        print(results["confusion_matrix"])
