import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load the data
train = pd.read_csv("SVM/data/bank-note/train.csv")
test = pd.read_csv("SVM/data/bank-note/test.csv")

# Extract features and labels
x_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values
x_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values

# Convert labels to +1 and -1
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Kernel (linear kernel in this case)
def linear_kernel(X, Y):
    return np.dot(X, Y.T)

# Define the dual objective function
def dual_objective(alpha, X, y, C):
    # Compute the kernel matrix
    K = linear_kernel(X, X)
    # Compute the dual objective (Lagrangian)
    return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)

# Define the constraint (sum of alpha_i * y_i = 0)
def constraint(alpha, y):
    return np.dot(alpha, y)

# SVM Dual Solver using `scipy.optimize.minimize`
def svm_dual(X, y, C):
    n_samples = X.shape[0]
    # Initial guess for alpha
    alpha_init = np.zeros(n_samples)
    
    # Bounds for the alpha values (0 <= alpha_i <= C)
    bounds = [(0, C) for _ in range(n_samples)]
    
    # Constraint for the sum of alpha_i * y_i = 0
    cons = {'type': 'eq', 'fun': constraint, 'args': (y,)}
    
    # Solve the optimization problem
    result = minimize(dual_objective, alpha_init, args=(X, y, C), bounds=bounds, constraints=cons, method='SLSQP')
    
    # Optimal alphas
    alpha_opt = result.x
    
    # Compute the weight vector w
    w = np.sum(alpha_opt[:, None] * y[:, None] * X, axis=0)
    
    # Compute the bias b
    support_vectors = alpha_opt > 1e-5  # Support vectors are those with alpha > 0
    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))
    
    return w, b, alpha_opt

# Run the dual SVM for different C values
C_values = [100/873, 500/873, 700/873]
results_dual = {}

for C in C_values:
    print(f"Training Dual SVM with C={C}")
    w_dual, b_dual, alpha_dual = svm_dual(x_train, y_train, C)
    
    # Train and test error
    train_preds = np.sign(np.dot(x_train, w_dual) + b_dual)
    test_preds = np.sign(np.dot(x_test, w_dual) + b_dual)
    
    # Calculate training error
    train_error = np.mean(train_preds != y_train)
    test_error = np.mean(test_preds != y_test)
    
    results_dual[C] = {'train_error': train_error, 'test_error': test_error}
    
    print(f"C = {C}: Train Error = {train_error:.4f}, Test Error = {test_error:.4f}")
    print(f"Weight vector (w): {w_dual}")
    print(f"Bias (b): {b_dual}\n")

# Display results
print("\nResults for 3a: \n")
for C in C_values:
    print(f"C = {C}: Train Error = {results_dual[C]['train_error']:.4f}, Test Error = {results_dual[C]['test_error']:.4f}")
print("\n")
