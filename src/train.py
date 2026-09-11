
from pathlib import Path
from .data import Dataset
from .model import DenseLayer, MultilayerPerceptron


class ModelTrainer:
    """Build, train, save, and plot a sequential MLP."""

    @staticmethod
    def create_network(hidden_layers, input_features, seed=42):
        """Build ``input -> hidden sigmoid layers -> one sigmoid output``."""

        layer_units = list(hidden_layers) + [1]
        model = MultilayerPerceptron(seed=seed)
        for index, units in enumerate(layer_units):
            is_output_layer = index == len(layer_units) - 1
            activation = "sigmoid" if is_output_layer else "relu"
            model.add(DenseLayer(units, activation))
            model.layers[-1].build(input_features, rng=model.rng)
            input_features = units

        return model


    @staticmethod
    def run(train_data, test_data, hidden_layers, epochs, batch_size, learning_rate, model_out, plot_out, seed=42):
        """Build, train, evaluate, plot, and save a model."""
        training = Dataset.load_csv(train_data).cleanup()
        input_features = training.X.shape[1]
        model = ModelTrainer.create_network(hidden_layers, input_features, seed)

        # Keep the scaler created during the split step with the model.
        scaler_file = Path("output/scaler.json")
        if scaler_file.exists():
            model.scaler = Dataset.load_scaler(str(scaler_file))
        elif Path("data/data.csv").exists():
            raw_data = Dataset.load_csv("data/data.csv").cleanup()
            model.scaler = raw_data.fit_scaler()

        model.fit(training.X, training.y, epochs, batch_size, learning_rate)

        if test_data:
            testing = Dataset.load_csv(test_data).cleanup()
            results = model.evaluate(testing.X, testing.y)
            print(f"Test accuracy: {results['accuracy']:.3f}")

        if plot_out:
            ModelTrainer.plot_curves(model.history, plot_out)
            print(f"Saved plot to {plot_out}")

        print(f"Saving model to {model_out}")
        model.save(model_out)

        return model

    @staticmethod
    def plot_curves(history, save_path="output/learning_curves.png"):
        import matplotlib.pyplot as plt

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(history["loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(epochs, history["loss"], label="train")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Binary cross-entropy")

        axes[1].plot(epochs, history["accuracy"], label="train")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")

        axes[0].legend()
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(save_path, dpi=150)
        plt.close(figure)
