


class ModelTrainer:
    """Coordinates model building, training, saving, and learning curves."""
    @staticmethod
    def run(dataset_path, val_dataset_path, hidden_layers, epochs, batch_size, learning_rate, model_out, plot_out, seed):
        """Build network, train, save model, and plot curves."""
        print(f"Dataset Path: {dataset_path}, Validation Dataset Path: {val_dataset_path}, Hidden Layers: {hidden_layers}, Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Model Output: {model_out}, Plot Output: {plot_out}, Seed: {seed}")
        pass

    @staticmethod
    def plot_curves(history, save_path="output/learning_curves.png"):
        """Plot loss and accuracy learning curves."""
        pass

