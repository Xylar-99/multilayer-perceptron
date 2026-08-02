import pandas as pd
import numpy as np
import sys


EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_LAYER_SIZES = 30



def preprocess_data(file_path):
    """ prepocess the data and return features and labels """

    df = pd.read_csv(file_path, header=None)
    df = df.drop(columns=[0])
    target_col = None

    for col in df.columns:
        values = set(df[col].dropna().unique())

        if values == {"M", "B"}:
            target_col = col
            break

    if target_col is None:
        raise ValueError("No diagnosis column (M/B) found.")

    y = df[target_col].map({"B": 0, "M": 1})
    x = df.drop(columns=[target_col])

    return x, y



def relu(x):
    return np.maximum(0, x)


class NeuralNetwork:

    def __init__(self, layers ):
        self.layers = layers

        self.weights = []
        self.biases = []

        self.activations = []

        self.create_network()

    def create_network(self):
        self.weights = [ np.random.uniform(-1, 1, (self.layers[i], self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        
    def forward(self, X):
        # print(f"Input shape: {self.weights[0]}")
        for i in range(len(self.weights)):
            print(f"Layer {i + 1}:")
            print(f"Weight shape: {self.weights[i]}")
            print(f"Input shape: {X}")
            z = np.dot(X, self.weights[i]) + self.biases[i]
            a = relu(z)
            sys.exit(0)
            # self.activations.append(a)
            # X = a



    def backward(self, X, y):
        pass

    def update_parameters(self):
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        pass


def train(args):
    """ Train the model """

    x , y = preprocess_data(args.dataset)

    nn = NeuralNetwork([x.shape[1], HIDDEN_LAYER_SIZES , HIDDEN_LAYER_SIZES // 2, 1])
    nn.create_network()
    nn.forward(x)
        
