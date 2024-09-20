import pandas as pd
import math

# Helper function to count labels in the dataset. Don't need to change.
def label_counts(data):
   label_counts = data.iloc[:, -1].value_counts().to_dict()
#    label_counts = {}
#    #look at last column which is the targets
#    for i in data.iloc[:, -1]:
#       if i not in label_counts:
#          label_counts[i] = 0
#       label_counts[i] = label_counts[i] + 1
   return label_counts

# Entropy calculation function. Don't need to change.
def entropy(data):
   label_count = label_counts(data)
   total_num_labels = len(data)
   entropy = -1 * sum((count / total_num_labels) * math.log2((count / total_num_labels)) for count in label_count.values())
   return entropy

# Gini Index calculation function. Don't need to change.
def gini_index(data):
   label_count = label_counts(data)
   total_num_labels = len(data)
   gini = 1 - sum((count / total_num_labels) ** 2 for count in label_count.values())
   return gini

# Majority Error calculation function. Don't need to change.
def majority_error(data):
   label_count = label_counts(data)
   total_num_labels = len(data)
   majority_label = max(label_count.values())
   return 1 - (majority_label / total_num_labels)

#split the data into subsets where the it holds every unique value in each attribute name. Save each dataset as a value in a dict to it's attribute
def split_data(data, attribute_name):
    data_subset = {}
    for attr_value in data[attribute_name].unique():
        data_subset[attr_value] = data[data[attribute_name] == attr_value]
    return data_subset

# Information Gain calculation. Don't need to change this
def information_gain(data, attribute_name, criterion_func):
    total_entropy = criterion_func(data)
    splits = split_data(data, attribute_name)
    total_size = len(data)
    weighted_entropy = 0
    for subset_of_data in splits.values():
        subset_size = len(subset_of_data)
        #calculate the weigthed average of the attribute using its subset size * its entropy
        weighted_entropy += (subset_size / total_size) * criterion_func(subset_of_data)
    return total_entropy - weighted_entropy

# Choosing the best attribute based on the chosen criterion. Don't need to change.
def choose_best_attribute(data, attributes, criterion_func):
    best_gain = -1
    best_attr = None
    for attr_name in attributes:
        # call the information gain function on each attribute. The highest ig is the best attribute to split on.
        gain = information_gain(data, attr_name, criterion_func)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr_name
    return best_attr

# Building the decision tree recursively
def lets_build_a_tree_baby(data, attributes, criterion_func, depth=0, max_depth=None):
    labels = data.iloc[:, -1]
    
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if not attributes or (max_depth is not None and depth == max_depth):
        return get_most_common_label(labels)
    
    best_attr = choose_best_attribute(data, attributes, criterion_func)
    
    tree = {best_attr: {}}
    splits = split_data(data, best_attr)
    remaining_attributes = [attr for attr in attributes if attr != best_attr]
    
    for attr_value, subset in splits.items():
        tree[best_attr][attr_value] = lets_build_a_tree_baby(subset, remaining_attributes, criterion_func, depth + 1, max_depth)
    return tree

#broken. please help. update: works if it is a pandas dataframe. We in business.
def get_most_common_label(labels):
    labels = label_counts(pd.DataFrame(labels))
    return max(labels, key=labels.get)

#no change
def predict(tree, example):
    # Base Case: Return the Label (Leaf Node). If the tree isn't a dict, then reached a leaf node
    if not isinstance(tree, dict):
        return tree
    # Get the Attribute to Split On
    attr = next(iter(tree))
    # Get the Value of the Attribute from the Example
    value = example[attr]
    # Find the next Subtree Based on the Example's Attribute Value
    subtree = tree[attr].get(value)
    if subtree is None:
        return None 
    # recursivley call function until the last subtree(leaf node) is found which represents the prediciton
    return predict(subtree, example)

# Function to evaluate the decision tree accuracy
def accuracy(tree, data):
    correct = 0
    #iterrows( ) is a function that goes through every row,
    for _, row in data.iterrows():
        prediction = predict(tree, row)
        # predict the value, if the predicted value is equal to the actual value, then the correct counter goes up one
        if prediction == row.iloc[-1]: 
            correct += 1
    return correct / len(data)

# Prediction error function
def prediction_error(tree, data):
    acc = accuracy(tree, data)
    return 1 - acc

# Run decision trees with varying depths
def run_different_depths(train_data, test_data, attributes, depths):
    results = {
        'Depth': [],
        'Criterion': [],
        'Train Error': [],
        'Test Error': []
    }
    
    print("Problem 3a:")
    for depth in depths:
        print(f"\n Making Tree with max depth = {depth}")

        tree_info_gain = lets_build_a_tree_baby(train_data, attributes, entropy, max_depth=depth)
        train_error_ig = prediction_error(tree_info_gain, train_data)
        test_error_ig = prediction_error(tree_info_gain, test_data)

        tree_majority_error = lets_build_a_tree_baby(train_data, attributes, majority_error, max_depth=depth)
        train_error_me = prediction_error(tree_majority_error, train_data)
        test_error_me = prediction_error(tree_majority_error, test_data)

        tree_gini_index = lets_build_a_tree_baby(train_data, attributes, gini_index, max_depth=depth)
        train_error_gi = prediction_error(tree_gini_index, train_data)
        test_error_gi = prediction_error(tree_gini_index, test_data)

        results['Depth'].append(depth)
        results['Criterion'].append('Information Gain(Entropy)')
        results['Train Error'].append(train_error_ig)
        results['Test Error'].append(test_error_ig)

        results['Depth'].append(depth)
        results['Criterion'].append('Majority Error')
        results['Train Error'].append(train_error_me)
        results['Test Error'].append(test_error_me)

        results['Depth'].append(depth)
        results['Criterion'].append('Gini Index')
        results['Train Error'].append(train_error_gi)
        results['Test Error'].append(test_error_gi)

    return pd.DataFrame(results)

# Binarize numerical attributes based on median. If it's less then median, then 0, if greater than, then 1
def make_binary(data, numerical_attributes):
    for attr in numerical_attributes:
        median_value = data[attr].median()
        data[attr] = (data[attr] >= median_value).astype(int)
    return data

# make a function that runs
# def prepare_data(train_data, test_data, numerical_attributes):
#     train_data = binarize_numerical_features(train_data, numerical_attributes)
#     for attr in numerical_attributes:
#         median_value = train_data[attr].median()
#         test_data[attr] = (test_data[attr] >= median_value).astype(int)
#     return train_data, test_data

# Load the Bank Marketing dataset (train.csv, test.csv)
train_data = pd.read_csv('bank/train.csv', header=None)
test_data = pd.read_csv('bank/test.csv', header=None)

# Define column names from 'data-desc.txt'
train_data.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
                      'previous', 'poutcome', 'label']
test_data.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
                      'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
                      'previous', 'poutcome', 'label']

numerical_attributes = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Prepare data
#train_data, test_data = prepare_data(train_data, test_data, numerical_attributes)

train_data = make_binary(train_data, numerical_attributes= numerical_attributes)
test_data = make_binary(test_data, numerical_attributes= numerical_attributes)

# get the names of the dataset columns, or the attributes
attributes = list(train_data.columns[:-1])

depths = range(1, 17)
results= run_different_depths(train_data, test_data, attributes, depths)

print("\nPrediction Errors Table for 3:")
print(results)
