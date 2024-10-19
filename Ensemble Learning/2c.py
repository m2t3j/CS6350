import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed

# Ensure all cores are used
N_CORES = -1  # setting = to -1 will use all available cores

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, data, attributes):
        self.tree = self._build_tree(data, attributes)

    def _build_tree(self, data, attributes):
        labels = data.iloc[:, -1]
        if len(labels.unique()) == 1:
            return labels.iloc[0]
        if not attributes:
            return labels.mode()[0]
        best_attr = self._choose_best_attribute(data, attributes)
        tree = {best_attr: {}}
        for attr_value, subset in split_data(data, best_attr).items():
            remaining_attrs = [attr for attr in attributes if attr != best_attr]
            subtree = self._build_tree(subset, remaining_attrs)
            tree[best_attr][attr_value] = subtree
        return tree

    def _choose_best_attribute(self, data, attributes):
        best_gain = -1
        best_attr = None
        for attr in attributes:
            gain = information_gain(data, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr

    def predict(self, example):
        return predict(self.tree, example)

class BaggedTrees:
    def __init__(self, num_trees=500):
        self.num_trees = num_trees
        self.trees = []

    def fit(self, data, attributes):
        # Train trees in parallel across 14 cores
        self.trees = Parallel(n_jobs=N_CORES, backend='loky')(
            delayed(self._train_single_tree)(data, attributes) for _ in range(self.num_trees)
        )

    def _train_single_tree(self, data, attributes):
        sample_data = bootstrap_sample(data)
        tree = DecisionTree()
        tree.fit(sample_data, attributes)
        return tree

    def predict(self, data):
        # Predict in parallel using 14 cores
        predictions = Parallel(n_jobs=N_CORES, backend='loky')(
            delayed(self._predict_single_tree)(tree, data) for tree in self.trees
        )
        return np.array(predictions)

    def _predict_single_tree(self, tree, data):
        return np.array([tree.predict(row) for _, row in data.iterrows()])

def bootstrap_sample(data):
    """Create a bootstrap sample from the given dataset."""
    n = len(data)
    sample_indices = [random.randint(0, n - 1) for _ in range(n)]
    return data.iloc[sample_indices]

def split_data(data, attribute_name):
    return {attr_value: data[data[attribute_name] == attr_value] for attr_value in data[attribute_name].unique()}

def information_gain(data, attribute_name):
    total_entropy = entropy(data)
    splits = split_data(data, attribute_name)
    total_size = len(data)
    weighted_entropy = sum((len(subset) / total_size) * entropy(subset) for subset in splits.values())
    return total_entropy - weighted_entropy

def entropy(data):
    label_count = data.iloc[:, -1].value_counts().to_dict()
    total = len(data)
    return -sum((count / total) * np.log2(count / total) for count in label_count.values())

def predict(tree, example):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = example[attr]
    subtree = tree[attr].get(value)
    if subtree is None:
        return 0  # Default prediction if value is not found
    return predict(subtree, example)

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

train_data['label'] = train_data['label'].replace({'yes': 1, 'no': 0})
test_data['label'] = test_data['label'].replace({'yes': 1, 'no': 0})

X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Experiment: Run 100 times with bagged trees and single trees
num_repeats = 100
bagged_predictions = np.zeros((num_repeats, len(X_test)))
single_tree_predictions = np.zeros((num_repeats, len(X_test)))

for i in range(num_repeats):
    sample_data = train_data.sample(1000, replace=False)
    attributes = list(X_test.columns)

    # Train 500 bagged trees
    bagged_model = BaggedTrees(num_trees=500)
    bagged_model.fit(sample_data, attributes)
    bagged_predictions[i] = bagged_model.predict(X_test).mean(axis=0)

    # Use the first tree from the bagged model as a single tree learner
    single_tree = bagged_model.trees[0]
    single_tree_predictions[i] = [single_tree.predict(row) for _, row in X_test.iterrows()]

# Compute bias, variance, and squared error for single trees
bias_single = np.mean((np.mean(single_tree_predictions, axis=0) - y_test) ** 2)
variance_single = np.mean(np.var(single_tree_predictions, axis=0))
squared_error_single = bias_single + variance_single

# Compute bias, variance, and squared error for bagged trees
bias_bagged = np.mean((np.mean(bagged_predictions, axis=0) - y_test) ** 2)
variance_bagged = np.mean(np.var(bagged_predictions, axis=0))
squared_error_bagged = bias_bagged + variance_bagged

# Print the results
print(f"Single Tree - Bias: {bias_single}, Variance: {variance_single}, Squared Error: {squared_error_single}")
print(f"Bagged Trees - Bias: {bias_bagged}, Variance: {variance_bagged}, Squared Error: {squared_error_bagged}")

# Plot the results
labels = ['Bias', 'Variance', 'Squared Error']
single_values = [bias_single, variance_single, squared_error_single]
bagged_values = [bias_bagged, variance_bagged, squared_error_bagged]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, single_values, width, label='Single Tree')
ax.bar(x + width/2, bagged_values, width, label='Bagged Trees')

ax.set_xlabel('Metric')
ax.set_title('Bias, Variance, and Squared Error Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
