"""Build, train, evaluate, plot, and save the ReLU-plus-softmax MLP."""

from pathlib import Path

from .data import Dataset, load_scaler
from .model import DenseLayer, MultilayerPerceptron
# from .predict import print_evaluation





def plot_learning_curves(history, save_path):
    """Save categorical-loss and accuracy curves collected after each epoch."""
    import matplotlib.pyplot as plt

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    for axis, values, title, ylabel in (
        (axes[0], history["loss"], "Loss", "Categorical cross-entropy"),
        (axes[1], history["accuracy"], "Accuracy", "Accuracy"),
    ):
        axis.plot(epochs, values)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def train_model(train_data_path,test_data_path,hidden_layers,epochs,batch_size,learning_rate,model_out,plot_out,seed=42,scaler_path="scaler.pkl"):
    """Run the full training workflow and return the trained model."""
    training_data = Dataset.from_csv(train_data_path).clean()
    input_features = training_data.X.shape[1]

    model = MultilayerPerceptron(input_features, hidden_layers, seed)

    test_data = Dataset.from_csv(test_data_path).clean()

    model.fit(training_data.X, training_data.y, epochs, batch_size, learning_rate)



    # plot_learning_curves(model.history, plot_out)
    model.save(model_out)

    return model
