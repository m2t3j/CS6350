import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Decision stump implementation with weighted examples
class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y, weights):
        X = np.array(X)
        y = np.array(y, dtype=float)
        weights = np.array(weights, dtype=float)

        n_features = X.shape[1]
        best_gain = -1

        for feature in range(n_features):
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_class = np.sign(np.sum(weights[left_mask] * y[left_mask]))
                right_class = np.sign(np.sum(weights[right_mask] * y[right_mask]))

                weighted_error = np.sum(weights[left_mask] * (y[left_mask] != left_class)) + \
                                 np.sum(weights[right_mask] * (y[right_mask] != right_class))

                gain = 1 - weighted_error / np.sum(weights)
                if gain > best_gain:
                    best_gain = gain
                    self.feature = feature
                    self.threshold = threshold
                    self.left_class = left_class
                    self.right_class = right_class

    def predict(self, X):
        predictions = np.where(X[:, self.feature] <= self.threshold, self.left_class, self.right_class)
        return predictions

# AdaBoost implementation
class AdaBoost:
    def __init__(self, T=50):
        self.T = T
        self.stumps = []
        self.stump_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.T):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            predictions = stump.predict(X)

            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            stump_weight = 0.5 * np.log((1 - error) / (error + 1e-10))

            self.stumps.append(stump)
            self.stump_weights.append(stump_weight)

            weights *= np.exp(-stump_weight * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for stump, weight in zip(self.stumps, self.stump_weights):
            final_predictions += weight * stump.predict(X)
        return np.sign(final_predictions)

# Load and preprocess the dataset
train_data = pd.read_csv('Ensemble Learning/Data/bank/train.csv', header=None)
test_data = pd.read_csv('Ensemble Learning/Data/bank/test.csv', header=None)

train_data.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                      'previous', 'poutcome', 'label']
test_data.columns = train_data.columns

numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

def make_binary(data, numerical_attributes):
    for attr in numerical_attributes:
        median_value = data[attr].median()
        data[attr] = (data[attr] >= median_value).astype(int)
    return data

train_data = make_binary(train_data, numerical_attributes)
test_data = make_binary(test_data, numerical_attributes)

train_data['label'] = train_data['label'].replace({'yes': 1, 'no': -1})
test_data['label'] = test_data['label'].replace({'yes': 1, 'no': -1})

X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Collect train and test errors for each stump at every iteration
def fit_and_evaluate_stumps(T):
    model = AdaBoost(T=T)
    model.fit(X_train, y_train)

    stump_train_errors = []
    stump_test_errors = []

    for stump, weight in zip(model.stumps, model.stump_weights):
        stump_train_error = np.mean(stump.predict(X_train) != y_train)
        stump_test_error = np.mean(stump.predict(X_test) != y_test)
        stump_train_errors.append(stump_train_error)
        stump_test_errors.append(stump_test_error)

    train_error = np.mean(model.predict(X_train) != y_train)
    test_error = np.mean(model.predict(X_test) != y_test)

    return train_error, test_error, stump_train_errors, stump_test_errors

T_values = range(1, 501)

print("Calculating 2a Errors... Will take about 10 minutes")
# Use joblib's Parallel to parallelize over the T values
results = Parallel(n_jobs=-1)(delayed(fit_and_evaluate_stumps)(T) for T in T_values)

# Unpack the results
train_errors, test_errors, all_stump_train_errors, all_stump_test_errors = zip(*results)

print("Printing Plots :")

# Plot 1: Training and test errors for AdaBoost over iterations
plt.figure(figsize=(10, 5))
plt.plot(T_values, train_errors, label='Train Error')
plt.plot(T_values, test_errors, label='Test Error')
plt.xlabel('Iterations (T)')
plt.ylabel('Error')
plt.legend()
plt.title('AdaBoost Training and Test Error vs. Iterations')
plt.show()

# Plot 2: Training and test errors for individual stumps
plt.figure(figsize=(10, 5))
for i, (stump_train_errors, stump_test_errors) in enumerate(zip(all_stump_train_errors, all_stump_test_errors)):
    plt.plot([i + 1] * len(stump_train_errors), stump_train_errors, 'ro', alpha=0.5, label='Train Error' if i == 0 else "")
    plt.plot([i + 1] * len(stump_test_errors), stump_test_errors, 'bo', alpha=0.5, label='Test Error' if i == 0 else "")

plt.xlabel('Iteration (T)')
plt.ylabel('Error')
plt.title('Errors of Individual Decision Stumps Across Iterations')
plt.legend()
plt.show()

print(
    "2a. Generally, the test error for the Adaboost is about half of that "
    "of the decision tree from homework 1 and did better at generalization.\n"
    "The training error for the decision tree was practically 0, meaning it "
    "was overfitting, and the results of the Adaboost do not show that."
)