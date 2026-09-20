
import matplotlib.pyplot as plt
from .data import Dataset, load_scaler
from .model import DenseLayer, MultilayerPerceptron


def plot_learning_curves(history, output_path):
    """Plot training and validation loss and accuracy curves."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(
        epochs,
        history["loss"],
        label="Training Loss",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy",
    )
    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()
    plt.close(figure)



def train_model(train_data_path,test_data_path,hidden_layers,epochs,batch_size,learning_rate,model_out,plot_out,seed=42,scaler_path="scaler.pkl"):
    """Run the full training workflow and return the trained model."""
    
    training_data = Dataset.from_csv(train_data_path).clean()
    test_data = Dataset.from_csv(test_data_path).clean()
    
    input_features = training_data.X.shape[1]
    model = MultilayerPerceptron(input_features, hidden_layers, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed)

    model.fit(training_data.X, training_data.y , test_data.X, test_data.y)

    plot_learning_curves(model.history, plot_out)
    model.save(model_out)

    return model
