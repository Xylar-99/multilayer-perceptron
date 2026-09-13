from pathlib import Path

from .data import Dataset
from .model import DenseLayer, MultilayerPerceptron


class ModelTrainer:
    """Build, train, evaluate, save, and plot a sequential MLP."""

    @staticmethod
    def run(
        train_data,
        test_data,
        hidden_layers,
        epochs,
        batch_size,
        learning_rate,
        model_out,
        plot_out,
        seed=42,
        scaler_path=None,
    ):
        """Run the complete training workflow and return the trained model."""
        training_data = ModelTrainer._load_dataset(train_data)
        model = ModelTrainer._create_model(training_data, hidden_layers, seed)

        ModelTrainer._attach_scaler(model, scaler_path)
        ModelTrainer._train_model(
            model,
            training_data,
            epochs,
            batch_size,
            learning_rate,
        )
        # ModelTrainer._evaluate_test_data(model, test_data)
        # ModelTrainer._save_learning_curves(model.history, plot_out)
        # ModelTrainer._save_model(model, model_out)

        return model

    @staticmethod
    def create_network(hidden_layers, input_features, seed=42):
        """Build ``input -> ReLU hidden layers -> one sigmoid output``."""
        layer_sizes = list(hidden_layers) + [1]
        model = MultilayerPerceptron(seed=seed)
        current_input_features = input_features

        for layer_index, units in enumerate(layer_sizes):
            is_output_layer = layer_index == len(layer_sizes) - 1
            activation = "sigmoid" if is_output_layer else "relu"

            ModelTrainer._add_layer(
                model,
                current_input_features,
                units,
                activation,
            )
            current_input_features = units

        return model

    @staticmethod
    def plot_curves(history, save_path="output/learning_curves.png"):
        """Save side-by-side loss and accuracy curves from training history."""
        import matplotlib.pyplot as plt

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        epoch_numbers = range(1, len(history["loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))

        ModelTrainer._plot_metric(
            axes[0],
            epoch_numbers,
            history["loss"],
            title="Loss",
            ylabel="Binary cross-entropy",
        )
        ModelTrainer._plot_metric(
            axes[1],
            epoch_numbers,
            history["accuracy"],
            title="Accuracy",
            ylabel="Accuracy",
        )

        figure.tight_layout()
        figure.savefig(save_path, dpi=150)
        plt.close(figure)

    @staticmethod
    def _load_dataset(dataset_path):
        """Load a labeled CSV file and remove incomplete rows."""
        return Dataset.load_csv(dataset_path).cleanup()

    @staticmethod
    def _create_model(training_data, hidden_layers, seed):
        """Create a network whose input size matches the training data."""
        input_features = training_data.X.shape[1]
        return ModelTrainer.create_network(hidden_layers, input_features, seed)

    @staticmethod
    def _add_layer(model, input_features, units, activation):
        """Create, initialize, and add one fully connected layer."""
        layer = DenseLayer(units, activation)
        layer.build(input_features, rng=model.rng)
        model.add(layer)

    @staticmethod
    def _attach_scaler(model, scaler_path=None):
        """Store a train-set scaler in the model when one is available.

        An explicitly supplied path must exist. Without one, the standard
        split-step location is used only when that file already exists.
        """
        if scaler_path is not None:
            model.scaler = Dataset.load_scaler(scaler_path)
            return

        default_scaler_path = Path("output/scaler.json")
        if default_scaler_path.exists():
            model.scaler = Dataset.load_scaler(str(default_scaler_path))

    @staticmethod
    def _train_model(model, training_data, epochs, batch_size, learning_rate):
        """Train a model on the prepared training dataset."""
        model.fit(
            training_data.X,
            training_data.y,
            epochs,
            batch_size,
            learning_rate,
        )

    @staticmethod
    def _evaluate_test_data(model, test_data):
        """Evaluate and display accuracy when a test dataset was supplied."""
        if not test_data:
            return None

        testing_data = ModelTrainer._load_dataset(test_data)
        results = model.evaluate(testing_data.X, testing_data.y)
        print(f"Test accuracy: {results['accuracy']:.3f}")
        return results

    @staticmethod
    def _save_learning_curves(history, plot_out):
        """Save a plot only when the caller requested one."""
        if not plot_out:
            return

        ModelTrainer.plot_curves(history, plot_out)
        print(f"Saved plot to {plot_out}")

    @staticmethod
    def _save_model(model, model_out):
        """Save the trained model and show its destination."""
        print(f"Saving model to {model_out}")
        model.save(model_out)

    @staticmethod
    def _plot_metric(axis, epoch_numbers, values, title, ylabel):
        """Draw one training metric with consistent labels."""
        axis.plot(epoch_numbers, values, label="train")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.legend()
