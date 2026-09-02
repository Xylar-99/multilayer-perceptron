import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_config')
import argparse
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data import Dataset
from src.model import Dense, MultilayerPerceptron


def plot_curves(history, save_path="output/learning_curves.png"):
    epochs = range(1, len(history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if history.get('val_loss'):
        ax1.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Curve over Epochs', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(epochs, history['acc'], 'b-', label='Training Accuracy', linewidth=2)
    if history.get('val_acc'):
        ax2.plot(epochs, history['val_acc'], 'g--', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy Curve over Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"> Learning curves saved to '{save_path}'")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Multilayer Perceptron on Breast Cancer dataset.")
    parser.add_argument("--dataset", type=str, default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--val_dataset", type=str, default="data/test.csv", help="Path to validation CSV")
    parser.add_argument("--layer", type=int, nargs="+", default=[24, 24], help="Hidden layer sizes (default: 24 24)")
    parser.add_argument("--epochs", type=int, default=84, help="Epochs (default: 84)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=0.0314, help="Learning rate (default: 0.0314)")
    parser.add_argument("--activation", type=str, default="sigmoid", help="Hidden activation (sigmoid, relu, tanh)")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer (sgd, adam)")
    parser.add_argument("--loss", type=str, default="categoricalCrossentropy", help="Loss function")
    parser.add_argument("--model_out", type=str, default="output/saved_model.json", help="Model output path")
    parser.add_argument("--plot_out", type=str, default="output/learning_curves.png", help="Plot output path")
    parser.add_argument("--early_stopping", type=int, default=None, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    train_ds = Dataset.from_csv(args.dataset)
    val_ds = Dataset.from_csv(args.val_dataset) if (args.val_dataset and os.path.exists(args.val_dataset)) else None

    # Construct network
    layers = [Dense(units=u, activation=args.activation, initializer='heUniform') for u in args.layer]
    layers.append(Dense(units=2, activation='softmax', initializer='heUniform'))

    model = MultilayerPerceptron.create_network(layers, seed=args.seed)
    history = model.fit(
        train_data=train_ds, val_data=val_ds, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        optimizer=args.optimizer, loss=args.loss,
        early_stopping=args.early_stopping, verbose=True
    )

    model.save(args.model_out)
    if not args.no_plot:
        plot_curves(history, save_path=args.plot_out)


if __name__ == "__main__":
    main()
