import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_weights(input_size, hidden_sizes, output_size):
    weights = []
    biases = []

    # Input to first hidden layer
    weights.append(np.random.randn(hidden_sizes[0], input_size) * 0.01)
    biases.append(np.zeros((hidden_sizes[0], 1)))

    # Between hidden layers
    for i in range(1, len(hidden_sizes)):
        weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i-1]) * 0.01)
        biases.append(np.zeros((hidden_sizes[i], 1)))

    # Last hidden to output layer
    weights.append(np.random.randn(output_size, hidden_sizes[-1]) * 0.01)
    biases.append(np.zeros((output_size, 1)))

    return weights, biases

def forward_propagation(X, weights, biases):
    activations = [X]
    zs = []

    for W, b in zip(weights, biases):
        z = np.dot(W, activations[-1]) + b
        a = sigmoid(z)
        zs.append(z)
        activations.append(a)

    return activations, zs

def backward_propagation(X, y, weights, biases, activations, zs):
    gradients_w = [np.zeros_like(W) for W in weights]
    gradients_b = [np.zeros_like(b) for b in biases]

    # Output layer error
    delta = activations[-1] - y.reshape(-1, 1)
    gradients_w[-1] = np.dot(delta, activations[-2].T)
    gradients_b[-1] = delta

    # Backpropagate through hidden layers
    for l in range(len(weights) - 2, -1, -1):
        delta = np.dot(weights[l + 1].T, delta) * sigmoid_derivative(zs[l])
        gradients_w[l] = np.dot(delta, activations[l].T)
        gradients_b[l] = delta

    return gradients_w, gradients_b

# Read in data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    X_train = train_data.iloc[:, :-1].values.T  # Features (transpose to match input format)
    y_train = train_data.iloc[:, -1].values     # Labels

    X_test = test_data.iloc[:, :-1].values.T   # Features (transpose to match input format)
    y_test = test_data.iloc[:, -1].values      # Labels

    return X_train, y_train, X_test, y_test

# Example usage
np.random.seed(42)
input_size = 4
hidden_sizes = [5, 3]
output_size = 1

# Load data
train_path = "Neural Networks/bank-note/train.csv"
test_path = "Neural Networks/bank-note/test.csv"
X_train, y_train, X_test, y_test = load_data(train_path, test_path)

# Initialize weights and biases
weights, biases = initialize_weights(input_size, hidden_sizes, output_size)

# Forward propagation
activations, zs = forward_propagation(X_train[:, [0]], weights, biases)  # Single example

# Backward propagation
gradients_w, gradients_b = backward_propagation(X_train[:, [0]], y_train[[0]], weights, biases, activations, zs)

print("Weight gradients:", gradients_w)
print("Bias gradients:", gradients_b)
print("\n")
