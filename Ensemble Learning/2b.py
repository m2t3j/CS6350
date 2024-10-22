import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Custom implementation of a Decision Tree (limited depth for simplicity)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X, y, depth=0):
        # If all labels are the same or max depth is reached, stop splitting
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            self.label = np.sign(np.sum(y))  # Assign majority class (+1 or -1)
            return

        # Find the best split
        best_gain = -1
        n_features = X.shape[1]
        for feature in range(n_features):
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_class = np.sign(np.sum(y[left_mask]))
                right_class = np.sign(np.sum(y[right_mask]))

                weighted_error = (
                    np.sum(y[left_mask] != left_class) + np.sum(y[right_mask] != right_class)
                ) / len(y)

                gain = 1 - weighted_error
                if gain > best_gain:
                    best_gain = gain
                    self.feature = feature
                    self.threshold = threshold

        # Split the data
        left_mask = X[:, self.feature] <= self.threshold
        right_mask = ~left_mask

        # Recursively train left and right subtrees
        self.left = DecisionTree(max_depth=self.max_depth)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)

        self.right = DecisionTree(max_depth=self.max_depth)
        self.right.fit(X[right_mask], y[right_mask], depth + 1)

    def predict(self, X):
        if self.label is not None:
            return np.full(X.shape[0], self.label)

        left_mask = X[:, self.feature] <= self.threshold
        right_mask = ~left_mask

        predictions = np.empty(X.shape[0])
        predictions[left_mask] = self.left.predict(X[left_mask])
        predictions[right_mask] = self.right.predict(X[right_mask])
        return predictions

# Function to train a tree on a bootstrapped sample
def train_tree(X, y, max_depth=None):
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_sample = X[indices]
    y_sample = y[indices]

    tree = DecisionTree(max_depth=max_depth)
    tree.fit(X_sample, y_sample)
    return tree

# Function to perform majority voting on predictions from multiple trees
def majority_vote(trees, X):
    predictions = np.array([tree.predict(X) for tree in trees])
    return np.sign(np.sum(predictions, axis=0))

# Load and preprocess the dataset
train_data = pd.read_csv('Ensemble Learning/Data/bank/train.csv', header=None)
test_data = pd.read_csv('Ensemble Learning/Data/bank/test.csv', header=None)

train_data.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                      'previous', 'poutcome', 'label']
test_data.columns = train_data.columns

# Impute missing values (simple median for numerical, mode for categorical)
def impute_missing_values(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)
    return data

train_data = impute_missing_values(train_data)
test_data = impute_missing_values(test_data)

# Binarize numerical attributes
numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

def make_binary(data, numerical_attributes):
    for attr in numerical_attributes:
        median_value = data[attr].median()
        data[attr] = (data[attr] >= median_value).astype(int)
    return data

train_data = make_binary(train_data, numerical_attributes)
test_data = make_binary(test_data, numerical_attributes)

# Convert labels to +1 and -1
train_data['label'] = train_data['label'].replace({'yes': 1, 'no': -1})
test_data['label'] = test_data['label'].replace({'yes': 1, 'no': -1})

X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Train multiple trees in parallel
def train_bagged_trees(X, y, n_trees=50, max_depth=None):
    return Parallel(n_jobs=-1)(delayed(train_tree)(X, y, max_depth) for _ in range(n_trees))

# Train 100 Bagged Trees
print("Training Bagged Trees...")
trees = train_bagged_trees(X_train, y_train, n_trees=100, max_depth=5)

# Make predictions using majority voting
train_predictions = majority_vote(trees, X_train)
test_predictions = majority_vote(trees, X_test)

# Calculate train and test errors
train_error = np.mean(train_predictions != y_train)
test_error = np.mean(test_predictions != y_test)

print(f"Train Error: {train_error:.4f}")
print(f"Test Error: {test_error:.4f}")

# Plot the results
plt.bar(['Train Error', 'Test Error'], [train_error, test_error])
plt.title('Bagged Trees Performance')
plt.show()



print("2b: Generally, it seems that the Bagged Trees have less error(though only slightly) than the Adaboost model and have better efficency. The single tree from HW1 was the worst performing")
