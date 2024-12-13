import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_weights(input_size, hidden_sizes, output_size):
    weights = []
    biases = []

    # Input to first hidden layer
    weights.append(np.random.randn(hidden_sizes[0], input_size))
    biases.append(np.zeros((hidden_sizes[0], 1)))

    # Between hidden layers
    for i in range(1, len(hidden_sizes)):
        weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i-1]))
        biases.append(np.zeros((hidden_sizes[i], 1)))

    # Last hidden to output layer
    weights.append(np.random.randn(output_size, hidden_sizes[-1]))
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

def compute_loss(y, y_pred):
    return 0.5 * np.mean((y - y_pred.flatten()) ** 2)

def stochastic_gradient_descent(X_train, y_train, X_test, y_test, input_size, output_size, hidden_widths, gamma0, d, epochs):
    training_errors = {}
    test_errors = {}

    for width in hidden_widths:
        hidden_sizes = [width, width]
        weights, biases = initialize_weights(input_size, hidden_sizes, output_size)

        losses = []

        for epoch in range(epochs):
            indices = np.arange(X_train.shape[1])
            np.random.shuffle(indices)

            for t, i in enumerate(indices, start=1):
                X_sample = X_train[:, [i]]
                y_sample = y_train[[i]]

                activations, zs = forward_propagation(X_sample, weights, biases)
                gradients_w, gradients_b = backward_propagation(X_sample, y_sample, weights, biases, activations, zs)

                learning_rate = gamma0 / (1 + gamma0 * t / d)

                for l in range(len(weights)):
                    weights[l] -= learning_rate * gradients_w[l]
                    biases[l] -= learning_rate * gradients_b[l]

            # Compute loss after each epoch
            y_pred_train = forward_propagation(X_train, weights, biases)[0][-1]
            y_pred_test = forward_propagation(X_test, weights, biases)[0][-1]
            losses.append(compute_loss(y_train, y_pred_train))

        training_errors[width] = compute_loss(y_train, forward_propagation(X_train, weights, biases)[0][-1])
        test_errors[width] = compute_loss(y_test, forward_propagation(X_test, weights, biases)[0][-1])

        # Plot convergence curve
        plt.plot(range(epochs), losses, label=f"Width {width}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence of Objective Function")
    plt.legend()
    plt.show()

    return training_errors, test_errors

np.random.seed(42)
train_path = "Neural Networks/bank-note/train.csv"
test_path = "Neural Networks/bank-note/test.csv"
X_train, y_train, X_test, y_test = load_data(train_path, test_path)

input_size = X_train.shape[0]
output_size = 1
hidden_widths = [5, 10, 25, 50, 100]
gamma0 = 0.1
d = 10
epochs = 50

training_errors, test_errors = stochastic_gradient_descent(X_train, y_train, X_test, y_test, input_size, output_size, hidden_widths, gamma0, d, epochs)

print("Training Errors:", training_errors)
print("Test Errors:", test_errors)
print('\n')
