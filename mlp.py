import argparse
from src.split import split
from src.train import train


def predict(args):
    print("Predict")
    print(args.dataset)
    print(args.model)


def create_parser():
    parser = argparse.ArgumentParser(description="Multilayer Perceptron")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )


    # ---------------- split ----------------

    split_parser = subparsers.add_parser("split", help="Split dataset" )

    split_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset path"
    )

    split_parser.add_argument(
        "--percent_train",
        type=int,
        default=80,
        help="Percentage of training data"
    )
    split_parser.add_argument(
        "--train_path",
        default="./data/train.csv",
        help="Path to save training data"
    )

    split_parser.add_argument(
        "--test_path",
        default="./data/test.csv",
        help="Path to save testing data"
    )

    split_parser.set_defaults(func=split)

    # ---------------- train ----------------

    train_parser = subparsers.add_parser("train", help="Train model")

    train_parser.add_argument(
        "--dataset",
        required=True
    )

    train_parser.set_defaults(func=train)

    # ---------------- predict ----------------

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict data"
    )

    predict_parser.add_argument(
        "--dataset",
        required=True
    )

    predict_parser.set_defaults(func=predict)

    return parser



def main():
    parser = create_parser()
    args = parser.parse_args()
    
    args.func(args)



if __name__ == "__main__":
    main()