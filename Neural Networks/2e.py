import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data
def load_data_torch(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None).values
    test_data = pd.read_csv(test_path, header=None).values

    X_train = torch.tensor(train_data[:, :-1], dtype=torch.float32)
    y_train = torch.tensor(train_data[:, -1], dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(test_data[:, :-1], dtype=torch.float32)
    y_test = torch.tensor(test_data[:, -1], dtype=torch.float32).view(-1, 1)

    return X_train, y_train, X_test, y_test

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_width, depth, activation):
        super(NeuralNetwork, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_width))
        layers.append(activation())

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_width, output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Xavier initialization
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# He initialization
def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test, input_size, output_size, hidden_widths, depths, activation, init_fn):
    training_errors = {}
    test_errors = {}

    for depth in depths:
        for width in hidden_widths:
            model = NeuralNetwork(input_size, output_size, width, depth, activation)
            model.apply(init_fn)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

            # Training
            for epoch in range(50):
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(X_train), y_train).item()
                test_loss = criterion(model(X_test), y_test).item()

            training_errors[(depth, width)] = train_loss
            test_errors[(depth, width)] = test_loss

    return training_errors, test_errors


train_path = "Neural Networks/bank-note/train.csv"
test_path = "Neural Networks/bank-note/test.csv"
X_train, y_train, X_test, y_test = load_data_torch(train_path, test_path)

input_size = X_train.shape[1]
output_size = 1
hidden_widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]

# Tanh activation with Xavier initialization
training_errors_tanh, test_errors_tanh = train_and_evaluate(
    X_train, y_train, X_test, y_test, input_size, output_size, hidden_widths, depths, nn.Tanh, xavier_init
)
print("Training Errors with Tanh:", training_errors_tanh)
print("Test Errors with Tanh:", test_errors_tanh)

# ReLU activation with He initialization
training_errors_relu, test_errors_relu = train_and_evaluate(
    X_train, y_train, X_test, y_test, input_size, output_size, hidden_widths, depths, nn.ReLU, he_init
)

print("\n")
print("Training Errors with ReLU:", training_errors_relu)
print("\n")
print("Test Errors with ReLU:", test_errors_relu)
print("\n")
print("2e answer: The ReLU activation with He initialization consistently achieved lower errors(though both performed really well) and faster convergence compared to Tanh with Xavier initialization. It had better performance than the other models above, probably due to the built in Adam optimizer.")
print("\n")