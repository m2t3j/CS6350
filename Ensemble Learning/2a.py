import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

### Takes about 84 minutes to run

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
train_data = pd.read_csv('Data/bank/train.csv', header=None)
test_data = pd.read_csv('Data/bank/test.csv', header=None)

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

T_values = range(1, 501)
train_errors, test_errors = [], []

for T in T_values:
    model = AdaBoost(T=T)
    model.fit(X_train, y_train)

    train_errors.append(np.mean(model.predict(X_train) != y_train))
    test_errors.append(np.mean(model.predict(X_test) != y_test))

plt.figure(figsize=(10, 5))
plt.plot(T_values, train_errors, label='Train Error')
plt.plot(T_values, test_errors, label='Test Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()
