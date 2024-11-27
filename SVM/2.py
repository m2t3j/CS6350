import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("SVM/data/bank-note/train.csv")
test = pd.read_csv("SVM/data/bank-note/test.csv")

x_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values
x_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values

# SVM uses +1 and -1 as labels
y_test = np.where(y_test == 0, -1, 1)
y_train = np.where(y_train == 0, -1, 1)

# Global Parameters
C_values = [100/873, 500/873, 700/873]
initial_learning_rate = .1
lr_decay = .01

# Objective/Loss function
def loss_function_SVM(y, X, w, b, C):
    hinge_loss = np.maximum(0, 1 - y * (np.dot(X, w) + b))
    return 0.5 * np.dot(w, w) + C * np.sum(hinge_loss)

def svm_primal_sgd(X, y, C, initial_learning_rate, lr_decay, max_epochs=100):
    # Initialize weights and bias
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    objective_values = []

    for epoch in range(max_epochs):
        # Shuffle the data
        index = np.random.permutation(n_samples)
        X_shuffle = X[index]
        y_shuffle = y[index]

        # Update the learning rate
        lr_update = initial_learning_rate / (1 + initial_learning_rate * lr_decay * epoch)

        # Algorithm - go through each point and update w, b
        for i in range(n_samples):
            x_point = X_shuffle[i]
            label = y_shuffle[i]
            decision_value = np.dot(w, x_point) + b

            # Misclassification case
            if label * decision_value < 1:
                w = w - lr_update * (w - C * label * x_point)
                b = b - lr_update * (-C * label)
            else:
                # Update weights (no misclassification)
                w = w - lr_update * w

        # Compute the objective value after the epoch
        obj_value = loss_function_SVM(y, X, w, b, C)
        objective_values.append(obj_value)

    return w, b, objective_values

def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

# Train and test the SVM with different C values
results = {}

for C in C_values:
    print(f"Training with C={C}")
    w, b, objective_values = svm_primal_sgd(X=x_train, y=y_train, C=C, initial_learning_rate=initial_learning_rate, lr_decay=lr_decay)
    
    # Plot the convergence curve (optional)
    plt.plot(objective_values)
    plt.title(f"Objective function curve (C={C})")
    plt.xlabel('Epoch')
    plt.ylabel('Objective Value')
    plt.show()

    # Train and test error
    train_preds = predict(x_train, w, b)
    test_preds = predict(x_test, w, b)

    # Calculate training error
    train_error = np.mean(train_preds != y_train)
    test_error = np.mean(test_preds != y_test)

    results[C] = {'train_error': train_error, 'test_error': test_error}

# Display results
print("\nResults for 2a: \n")
print("Weights: ", w ,"\n")
print("Bias: ", b ,"\n")
for C in C_values:
    print(f"C = {C}: Train Error = {results[C]['train_error']:.4f}, Test Error = {results[C]['test_error']:.4f}")
print("\n")


#### PART B

def svm_primal_sgd_b(X, y, C, initial_learning_rate, max_epochs=100):
    # Initialize weights and bias
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    objective_values_b = []

    for epoch in range(max_epochs):
        # Shuffle the data
        index = np.random.permutation(n_samples)
        X_shuffle = X[index]
        y_shuffle = y[index]

        # Update the learning rate with the new schedule
        lr_update = initial_learning_rate / (1 + epoch)

        # Algorithm - go through each point and update w, b
        for i in range(n_samples):
            x_point = X_shuffle[i]
            label = y_shuffle[i]
            decision_value = np.dot(w, x_point) + b

            # Misclassification case
            if label * decision_value < 1:
                w = w - lr_update * (w - C * label * x_point)
                b = b - lr_update * (-C * label)
            else:
                # Update weights (no misclassification)
                w = w - lr_update * w

        # Compute the objective value after the epoch
        obj_value = loss_function_SVM(y, X, w, b, C)
        objective_values_b.append(obj_value)

    return w, b, objective_values_b

results = {}

for C in C_values:
    print(f"Training with C={C}")
    w, b, objective_values_b = svm_primal_sgd_b(X=x_train, y=y_train, C=C, initial_learning_rate=initial_learning_rate)
    
    # Plot the convergence curve (optional)
    plt.plot(objective_values_b)
    plt.title(f"Objective function curve (C={C})")
    plt.xlabel('Epoch')
    plt.ylabel('Objective Value')
    plt.show()

    # Train and test error
    train_preds = predict(x_train, w, b)
    test_preds = predict(x_test, w, b)

    # Calculate training error
    train_error = np.mean(train_preds != y_train)
    test_error = np.mean(test_preds != y_test)

    results[C] = {'train_error': train_error, 'test_error': test_error}

# Display results
print("\nResults for 2b: \n")
print("Weights: ", w ,"\n")
print("Bias: ", b ,"\n")
for C in C_values:
    print(f"C = {C}: Train Error = {results[C]['train_error']:.4f}, Test Error = {results[C]['test_error']:.4f}")
print("\n")

### Part C: Comparison