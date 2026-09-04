


class ModelTrainer:
    """Coordinates model building, training, saving, and learning curves."""
    @staticmethod
    def run(train_data, test_data, hidden_layers, epochs, batch_size, learning_rate, model_out, plot_out):
        """Build network, train, save model, and plot curves."""
        print(f"Train Data: {train_data}, Test Data: {test_data}, Hidden Layers: {hidden_layers}, Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Model Output: {model_out}, Plot Output: {plot_out}")
        pass

    @staticmethod
    def plot_curves(history, save_path="output/learning_curves.png"):
        """Plot loss and accuracy learning curves."""
        pass

