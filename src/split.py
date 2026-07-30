import pandas as pd





def split(args):
    """ split the dataset into training and testing sets """

    dataset_path = args.dataset
    percent_train = args.percent_train
    train_path = args.train_path
    test_path = args.test_path

    # Load the dataset
    dataset = pd.read_csv(dataset_path)

    # Shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)



    # Split the dataset into training and testing sets
    train_ratio = percent_train / 100
    train_size = int(len(dataset) * train_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]


    # Save the training and testing sets to CSV files
    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path, index=False)

    print(f"Dataset split into {len(train_set)} training samples and {len(test_set)} testing samples.")