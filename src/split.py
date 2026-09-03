from .data import Dataset



class DataSplitter:

    @staticmethod
    def preprocess(dataset_path):
        print(f"Preprocessing dataset at: {dataset_path}")

        dataset = Dataset.load_csv(dataset_path)

        dataset.cleanup()

        return dataset

    @staticmethod
    def run(dataset_path, train_out, test_out, ratio, seed):

        # 1. Load + cleanup
        dataset = DataSplitter.preprocess(dataset_path)

        # 2. Split
        train_data, test_data = dataset.split(
            train_ratio=ratio,
            seed=seed
        )

        # 3. Fit scaler ONLY on train
        scaler = train_data.fit_scaler()

        # 4. Scale train and test using SAME scaler
        train_data.scale(scaler)
        test_data.scale(scaler)

        # 5. Save
        train_data.save_csv(train_out)
        test_data.save_csv(test_out)