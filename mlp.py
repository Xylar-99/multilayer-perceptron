import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_config')
import sys
import argparse

from src.data import Dataset
from src.model import Dense, MultilayerPerceptron
from src.train import plot_curves
from src.predict import print_report


def build_parser():
    parser = argparse.ArgumentParser(
        description="Multilayer Perceptron (MLP) — Breast Cancer Classification",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Modes
    mode_grp = parser.add_argument_group("Execution Modes")
    mode_grp.add_argument("--split", action="store_true", help="Split dataset into train and test sets")
    mode_grp.add_argument("--train", action="store_true", help="Train the neural network")
    mode_grp.add_argument("--predict", action="store_true", help="Evaluate predictions on a dataset")
    mode_grp.add_argument("mode", nargs="?", choices=["split", "train", "predict"], default=None,
                          help="Positional mode: 'split', 'train', or 'predict'")

    # Data options
    data_grp = parser.add_argument_group("Data Options")
    data_grp.add_argument("--dataset", type=str, default=None, help="Path to CSV dataset")
    data_grp.add_argument("--val_dataset", type=str, default="data/test.csv", help="Path to validation CSV")
    data_grp.add_argument("--train_out", type=str, default="data/train.csv", help="Destination train split CSV")
    data_grp.add_argument("--test_out", type=str, default="data/test.csv", help="Destination test split CSV")
    data_grp.add_argument("--ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")

    # Hyperparameters
    hyp_grp = parser.add_argument_group("Hyperparameters")
    hyp_grp.add_argument("--layer", type=int, nargs="+", default=[24, 24], help="Hidden layers (default: 24 24)")
    hyp_grp.add_argument("--epochs", type=int, default=84, help="Epochs (default: 84)")
    hyp_grp.add_argument("--batch_size", type=int, default=8, help="Mini-batch size (default: 8)")
    hyp_grp.add_argument("--learning_rate", type=float, default=0.0314, help="Learning rate (default: 0.0314)")
    hyp_grp.add_argument("--activation", type=str, default="sigmoid", help="Hidden activation (sigmoid, relu, tanh)")
    hyp_grp.add_argument("--optimizer", type=str, default="sgd", help="Optimizer (sgd, adam)")
    hyp_grp.add_argument("--loss", type=str, default="categoricalCrossentropy", help="Loss function")
    hyp_grp.add_argument("--early_stopping", type=int, default=None, help="Early stopping patience")
    hyp_grp.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output options
    io_grp = parser.add_argument_group("Output Options")
    io_grp.add_argument("--model", type=str, default="output/saved_model.json", help="Model file path")
    io_grp.add_argument("--model_out", type=str, default="output/saved_model.json", help="Model output path")
    io_grp.add_argument("--plot_out", type=str, default="output/learning_curves.png", help="Plot output path")
    io_grp.add_argument("--no_plot", action="store_true", help="Disable plotting")

    return parser


def run_split(dataset, train_out, test_out, ratio, seed):
    ds = Dataset.from_csv(dataset)
    train_ds, test_ds = ds.split(train_ratio=ratio, seed=seed)
    train_ds.save_csv(train_out)
    test_ds.save_csv(test_out)
    total = len(ds)
    print(f"Dataset split complete:")
    print(f"  Total samples: {total}")
    print(f"  Train set:     {len(train_ds)} samples ({len(train_ds)/total*100:.1f}%) -> {train_out}")
    print(f"  Test set:      {len(test_ds)} samples ({len(test_ds)/total*100:.1f}%) -> {test_out}")


def run_predict(dataset, model_path):
    if not os.path.exists(dataset):
        print(f"Error: Dataset '{dataset}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"> Loading model from '{model_path}'...")
    model = MultilayerPerceptron.load(model_path)
    print(f"> Loading dataset from '{dataset}'...")
    ds = Dataset.from_csv(dataset)
    res = model.evaluate(ds.X, ds.y)
    print(f"\nEvaluation on {len(ds)} samples:")
    print(f"Binary Cross-Entropy Loss : {res['bce_loss']:.6f}")
    print(f"Accuracy                  : {res['accuracy'] * 100:.2f}%")
    print_report(res)


def run_train(dataset, val_dataset, layers_cfg, epochs, batch_size, lr, activation, optimizer, loss, model_out, plot_out, early_stopping, seed, no_plot):
    train_ds = Dataset.from_csv(dataset)
    val_ds = Dataset.from_csv(val_dataset) if (val_dataset and os.path.exists(val_dataset)) else None

    layers = [Dense(units=u, activation=activation, initializer='heUniform') for u in layers_cfg]
    layers.append(Dense(units=2, activation='softmax', initializer='heUniform'))

    model = MultilayerPerceptron.create_network(layers, seed=seed)
    history = model.fit(
        train_data=train_ds, val_data=val_ds, epochs=epochs,
        batch_size=batch_size, learning_rate=lr,
        optimizer=optimizer, loss=loss,
        early_stopping=early_stopping, verbose=True
    )

    model.save(model_out)
    if not no_plot:
        plot_curves(history, save_path=plot_out)


def main():
    parser = build_parser()
    args = parser.parse_args()

    mode = "train"
    if args.split or args.mode == "split":
        mode = "split"
    elif args.predict or args.mode == "predict":
        mode = "predict"
    elif args.train or args.mode == "train":
        mode = "train"
    elif args.dataset and "test" in args.dataset:
        mode = "predict"

    if mode == "split":
        dataset = args.dataset or "data/data.csv"
        run_split(dataset, args.train_out, args.test_out, args.ratio, args.seed)
    elif mode == "predict":
        dataset = args.dataset or "data/test.csv"
        model_path = args.model if args.model != "output/saved_model.json" else args.model_out
        run_predict(dataset, model_path)
    else:
        dataset = args.dataset
        val_dataset = args.val_dataset
        if dataset is None:
            if not os.path.exists("data/train.csv") and os.path.exists("data/data.csv"):
                print("> Auto-splitting 'data/data.csv'...")
                run_split("data/data.csv", "data/train.csv", "data/test.csv", args.ratio, args.seed)
            dataset = "data/train.csv"

        if val_dataset and not os.path.exists(val_dataset):
            val_dataset = None

        model_path = args.model_out if args.model_out != "output/saved_model.json" else args.model
        run_train(
            dataset=dataset, val_dataset=val_dataset, layers_cfg=args.layer,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate,
            activation=args.activation, optimizer=args.optimizer, loss=args.loss,
            model_out=model_path, plot_out=args.plot_out, early_stopping=args.early_stopping,
            seed=args.seed, no_plot=args.no_plot
        )


if __name__ == "__main__":
    main()
