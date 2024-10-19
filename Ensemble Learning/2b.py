#b using parrallel computing

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

# Function to impute missing values with the majority value
def impute_missing_values(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            mode_value = data[column].mode()[0]
            data[column].fillna(mode_value, inplace=True)
        else:
            median_value = data[column].median()
            data[column].fillna(median_value, inplace=True)
    return data

# Load and preprocess the dataset
train_data = pd.read_csv('Ensemble Learning/Data/bank/train.csv', header=None)
test_data = pd.read_csv('Ensemble Learning/Data/bank/test.csv', header=None)

train_data.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                      'previous', 'poutcome', 'label']
test_data.columns = train_data.columns

# Impute missing values
train_data = impute_missing_values(train_data)
test_data = impute_missing_values(test_data)

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

# Parallelized function for fitting the AdaBoost model and collecting errors
def compute_errors(T):
    model = AdaBoost(T=T)
    model.fit(X_train, y_train)
    train_error = np.mean(model.predict(X_train) != y_train)
    test_error = np.mean(model.predict(X_test) != y_test)
    return train_error, test_error


print("Calculating 2b Errors: Will take about 10 minutes")
# Parallel execution over multiple iterations using joblib
T_values = range(1, 501)
results = Parallel(n_jobs=-1)(delayed(compute_errors)(T) for T in T_values)

# Extract train and test errors from the results
train_errors, test_errors = zip(*results)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(T_values, train_errors, label='Train Error')
plt.plot(T_values, test_errors, label='Test Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()


print("2b: Generally, it seems that the Bagged Trees have less error(though only slightly) than the Adaboost model and have better efficency. The single tree from HW1 was the worst performing")