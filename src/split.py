import argparse
from src.data import Dataset


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("--dataset", type=str, default="data/data.csv", help="Path to raw CSV dataset")
    parser.add_argument("--train_out", type=str, default="data/train.csv", help="Destination train split CSV")
    parser.add_argument("--test_out", type=str, default="data/test.csv", help="Destination test split CSV")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable shuffle")
    args = parser.parse_args()

    ds = Dataset.from_csv(args.dataset)
    train_ds, test_ds = ds.split(train_ratio=args.ratio, seed=args.seed)
    train_ds.save_csv(args.train_out)
    test_ds.save_csv(args.test_out)

    total = len(ds)
    print(f"Dataset split complete:")
    print(f"  Total samples: {total}")
    print(f"  Train set:     {len(train_ds)} samples ({len(train_ds)/total*100:.1f}%) -> {args.train_out}")
    print(f"  Test set:      {len(test_ds)} samples ({len(test_ds)/total*100:.1f}%) -> {args.test_out}")


if __name__ == "__main__":
    main()
