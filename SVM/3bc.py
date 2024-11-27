import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 

# Load the data
train = pd.read_csv("SVM/data/bank-note/train.csv")
test = pd.read_csv("SVM/data/bank-note/test.csv")
x_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values
x_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values


y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Gaussian Kernel 
def gaussian_kernel(X, Y, gamma):
    # Compute pairwise squared distances
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
    return np.exp(-sq_dists / gamma)

# Dual SVM Objective Function with Gaussian Kernel
def dual_objective(alpha, X, y, C, gamma):
    K = gaussian_kernel(X, X, gamma)
    return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)

# Constraint for the sum of alpha_i * y_i = 0
def constraint(alpha, y):
    return np.dot(alpha, y)

# SVM Dual Solver using `scipy.optimize.minimize`
def svm_dual(X, y, C, gamma):
    n_samples = X.shape[0]
    # Initial guess for alpha
    alpha_init = np.zeros(n_samples)
    
    # Bounds for the alpha values (0 <= alpha_i <= C)
    bounds = [(0, C) for _ in range(n_samples)]
    
    # Constraint for the sum of alpha_i * y_i = 0
    cons = {'type': 'eq', 'fun': constraint, 'args': (y,)}
    
    # Solve the optimization problem
    result = minimize(dual_objective, alpha_init, args=(X, y, C, gamma), bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 5000, 'disp': True})
    
    # Optimal alphas
    alpha_opt = result.x
    
    # Compute the weight vector w (from the dual)
    w = np.sum(alpha_opt[:, None] * y[:, None] * X, axis=0)
    
    # Compute the bias b
    support_vectors = alpha_opt > 1e-5  # Support vectors are those with alpha > 0
    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))
    
    return w, b, alpha_opt, support_vectors, result

# Prediction using the kernel
def predict(X, X_train, y_train, w, b, gamma):
    K = gaussian_kernel(X_train, X, gamma)
    return np.sign(np.dot(K.T, y_train) + b)

# Test different values of C and gamma
# already ran 100/873, change after it is done
C_values = [ 500/873]
#C_values = [100/873,500/873, 700/873]
gamma_values = [0.1, 0.5, 1, 5, 100]
results_dual = {}
support_vectors_per_gamma = {}

# Start tracking the time
start_time = time.time()

for C in C_values:
    for gamma in gamma_values:
        print(f"Training Dual SVM with C={C} and gamma={gamma}")
        
        # Track the start time for each iteration (to track the time elapsed for each combination)
        iteration_start_time = time.time()
        
        w_dual, b_dual, alpha_dual, support_vectors, result = svm_dual(x_train, y_train, C, gamma)
        
        # Calculate elapsed time for this iteration
        elapsed_time = (time.time() - iteration_start_time) / 60  # Convert seconds to minutes
        print(f"Time elapsed for C={C}, gamma={gamma}: {elapsed_time:.2f} minutes")
        
        # Track the number of support vectors
        num_support_vectors = np.sum(support_vectors)
        support_vectors_per_gamma[(C, gamma)] = num_support_vectors
        print(f"Number of support vectors for C={C}, gamma={gamma}: {num_support_vectors}")
        
        # Train and test error
        train_preds = predict(x_train, x_train, y_train, w_dual, b_dual, gamma)
        test_preds = predict(x_test, x_train, y_train, w_dual, b_dual, gamma)
        
        # Calculate training error
        train_error = np.mean(train_preds != y_train)
        test_error = np.mean(test_preds != y_test)
        
        results_dual[(C, gamma)] = {'train_error': train_error, 'test_error': test_error}
        
        print(f"C = {C}, gamma = {gamma}: Train Error = {train_error:.4f}, Test Error = {test_error:.4f}")

# Calculate and print total time elapsed
total_elapsed_time = (time.time() - start_time) / 60  # Convert seconds to minutes
print(f"\nTotal Time Elapsed: {total_elapsed_time:.2f} minutes")

# Display results
print("\nResults for 2b: \n")
for (C, gamma), errors in results_dual.items():
    print(f"C = {C}, gamma = {gamma}: Train Error = {errors['train_error']:.4f}, Test Error = {errors['test_error']:.4f}")

# Problem (c) - Report overlap of support vectors when C=500/873
if 500/873 in C_values:
    print("\nSupport Vectors Overlap for C=500/873 and Consecutive Gamma Values:")
    overlap_results = []
    gamma_combinations = [(0.1, 0.5), (0.5, 1), (1, 5), (5, 100)]
    
    for gamma1, gamma2 in gamma_combinations:
        # Get support vectors for each gamma value
        support_vectors_gamma1 = support_vectors_per_gamma.get((500/873, gamma1), [])
        support_vectors_gamma2 = support_vectors_per_gamma.get((500/873, gamma2), [])
        
        # Calculate overlap
        overlap = np.sum(support_vectors_gamma1 == support_vectors_gamma2)  # Count the common support vectors
        overlap_results.append((gamma1, gamma2, overlap))
        print(f"Overlap between gamma={gamma1} and gamma={gamma2}: {overlap} support vectors")
    
print("\n")



