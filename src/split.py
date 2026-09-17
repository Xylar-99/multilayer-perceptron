"""Prepare the raw dataset and save train/test files."""

import numpy as np

from .data import Dataset , save_scaler , scale_features


def split_dataset(dataset_path, train_out, test_out, scaler_out, ratio=0.8):
    """Clean, split, scale, and save the dataset."""
    dataset = Dataset.from_csv(dataset_path).clean()

    train_data, test_data = dataset.split(ratio)

    scaler = train_data.fit_scaler()

    train_data.scale(scaler)
    test_data.scale(scaler)

    train_data.save_csv(train_out)
    test_data.save_csv(test_out)

    save_scaler(scaler, scaler_out)

    print(f"x_train shape: {train_data.X.shape}")
    print(f"x_valid shape: {test_data.X.shape}")

    return train_data, test_data, scaler
