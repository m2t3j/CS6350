import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(over='warn')  # Set NumPy to warn on overflows

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    y_train = np.where(y_train == 0, -1, 1)  # Convert labels to {-1, 1}

    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    y_test = np.where(y_test == 0, -1, 1)  # Convert labels to {-1, 1}

    # Add bias term to the features
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    return X_train, y_train, X_test, y_test

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent overflow in exp
    return 1 / (1 + np.exp(-x))

def compute_objective_ml(X, y, w):
    # Logistic loss (without prior)
    log_likelihood = np.mean(np.logaddexp(0, -y * np.dot(X, w)))  # Stable computation of log(1 + exp(-ywx))
    return log_likelihood

def compute_gradient_ml(X, y, w):
    # Gradient of the logistic loss (without prior)
    predictions = sigmoid(np.dot(X, w))
    grad = -np.dot(X.T, y * (1 - predictions)) / X.shape[0]
    return grad

def stochastic_gradient_descent_ml(X_train, y_train, X_test, y_test, gamma0, d, epochs):
    results = {}
    w = np.zeros(X_train.shape[1])  # Initialize weights to zero
    losses = []

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        for t, i in enumerate(indices, start=1):
            X_sample = X_train[[i]]
            y_sample = y_train[[i]]
            grad = compute_gradient_ml(X_sample, y_sample, w)

            learning_rate = gamma0 / (1 + gamma0 * t / d)
            w -= learning_rate * grad

        # Compute loss after each epoch
        loss = compute_objective_ml(X_train, y_train, w)
        losses.append(loss)
        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    # Compute training and test errors
    train_predictions = np.sign(np.dot(X_train, w))
    test_predictions = np.sign(np.dot(X_test, w))
    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    # Plot convergence curve
    plt.plot(range(epochs), losses, label="Maximum Likelihood")
    plt.xlabel("Epoch")
    plt.ylabel("Objective Function")
    plt.title("Convergence of Objective Function (ML)")
    plt.legend()
    plt.show()

    results['train_error'] = train_error
    results['test_error'] = test_error
    results['losses'] = losses  

    return results

# File paths
train_path = "Neural Networks/bank-note/train.csv"
test_path = "Neural Networks/bank-note/test.csv"

# Load data
X_train, y_train, X_test, y_test = load_data(train_path, test_path)

# Parameters
gamma0 = 0.1
d = 10
epochs = 100

# Run SGD for ML estimation
results_ml = stochastic_gradient_descent_ml(X_train, y_train, X_test, y_test, gamma0, d, epochs)
print("Results (Training Error, Test Error):", results_ml)
print("\n")
