
from pathlib import Path
import matplotlib.pyplot as plt
from .data import Dataset
from .model import DenseLayer, MultilayerPerceptron


class ModelTrainer:
    """Build, train, save, and plot a sequential MLP."""
    @staticmethod
    def create_network(hidden_layers, input_features, seed=42):
        """Build ``input -> hidden sigmoid layers -> one sigmoid output``."""

        layer_units = list(hidden_layers) + [2]
        model = MultilayerPerceptron(seed=seed)
        for units in layer_units:
            activation = "softmax" if units == 2 else "relu"
            model.add(DenseLayer(units, activation))
            model.layers[-1].build(input_features, rng=model.rng)
            input_features = units

        return model


    @staticmethod
    def run(train_data, test_data, hidden_layers, epochs, batch_size, learning_rate, model_out, plot_out, seed=42):

        training = Dataset.load_csv(train_data).cleanup()
        input_features = training.X.shape[1]
        model = ModelTrainer.create_network(hidden_layers, input_features, seed=seed)

        A = model.forward(training.X)




        # Print predictions for each row in the training set
        for i, pred in enumerate(A):
            benign_prob = pred[0] * 100
            malignant_prob = pred[1] * 100
            print(f"Row {i+1} -> Benign: {benign_prob:.2f}% | Malignant: {malignant_prob:.2f}%")
        
        pass

    @staticmethod
    def plot_curves(history, save_path="output/learning_curves.png"):
        pass
        # Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # epochs = range(1, len(history["loss"]) + 1)
        # figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        # axes[0].plot(epochs, history["loss"], label="train")
        # axes[1].plot(epochs, history["acc"], label="train")
        # if history["val_loss"]:
        #     axes[0].plot(epochs, history["val_loss"], label="validation")
        #     axes[1].plot(epochs, history["val_acc"], label="validation")
        # axes[0].set(title="Binary cross-entropy", xlabel="Epoch", ylabel="Loss")
        # axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
        # axes[0].legend(); axes[1].legend()
        # figure.tight_layout(); figure.savefig(save_path, dpi=150); plt.close(figure)
