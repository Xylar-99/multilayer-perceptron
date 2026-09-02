import os
import sys
import argparse
from src.data import Dataset
from src.model import MultilayerPerceptron


def print_report(res, title="Test Set Evaluation"):
    cm = res['confusion_matrix']
    print(f"\n{'='*45}")
    print(f" {title.center(43)} ")
    print(f"{'='*45}")
    print(f"  Binary Cross-Entropy Loss: {res['bce_loss']:.4f}")
    print(f"  Accuracy:                  {res['accuracy'] * 100:.2f}% ({res['accuracy']:.4f})")
    print(f"  Precision (Malignant):     {res['precision'] * 100:.2f}% ({res['precision']:.4f})")
    print(f"  Recall (Malignant):        {res['recall'] * 100:.2f}% ({res['recall']:.4f})")
    print(f"  F1-Score:                  {res['f1_score'] * 100:.2f}% ({res['f1_score']:.4f})")
    print(f"{'-'*45}")
    print("  Confusion Matrix:")
    print(f"               Pred Benign   Pred Malignant")
    print(f"  True Benign       {cm[0,0]:<12} {cm[0,1]}")
    print(f"  True Malignant    {cm[1,0]:<12} {cm[1,1]}")
    print(f"{'='*45}\n")


def main():
    parser = argparse.ArgumentParser(description="Predict and evaluate Multilayer Perceptron on test dataset.")
    parser.add_argument("--dataset", type=str, default="data/test.csv", help="Path to evaluation CSV")
    parser.add_argument("--model", type=str, default="output/saved_model.json", help="Path to saved model")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset '{args.dataset}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"> Loading model from '{args.model}'...")
    model = MultilayerPerceptron.load(args.model)

    print(f"> Loading dataset from '{args.dataset}'...")
    ds = Dataset.from_csv(args.dataset)

    res = model.evaluate(ds.X, ds.y)
    print(f"\nEvaluation on {len(ds)} samples:")
    print(f"Binary Cross-Entropy Loss : {res['bce_loss']:.6f}")
    print(f"Accuracy                  : {res['accuracy'] * 100:.2f}%")
    print_report(res)


if __name__ == "__main__":
    main()
