import pandas as pd
import math

def label_counts(data):
   #label_counts = data.iloc[:-1].valuecounts()
   label_counts = {}
   #look at last column which is the targets
   for i in data.iloc[:, -1]:
      if i not in label_counts:
         label_counts[i] = 0
      label_counts[i] = label_counts[i] + 1
   return label_counts

### Entropy = - sum(p_i) * log2(p_i)
def entropy(data):
   label_count = label_counts(data)
   total_num_labels = len(data)
   entropy = -1*sum((count/total_num_labels)*math.log2((count/total_num_labels)) for count in label_count.values())
#    for i in label_counts.values():
#       prob = i / total_num_labels
#       entropy = -(prob * math.log2(prob))
   return entropy

### gini index = 1 - sum(p^2)
def gini_index(data):
   label_count = label_counts(data)
   total_num_lables = len(data)
   gini = 1 - sum((count / total_num_lables) ** 2 for count in label_count.values())
   return gini

### me = 1 - (majority class / total num)
def majority_error(data):
   label_count = label_counts(data)
   total_num_labels = len(data)
   majority_label = max(label_count.values())
   return 1 - (majority_label / total_num_labels)

#split the data into subsets where the it holds every unique value in each attribute name. Save each dataset as a value in a dict to it's attribute
def split_data(data, attribute_name):
    data_subset = {}
    # Loop through the unique values of the attribute
    for attr_value in data[attribute_name].unique():
        # Create a subset where the attribute equals the current value
        data_subset[attr_value] = data[data[attribute_name] == attr_value]
    return data_subset

# IG = total entropy - attribute weigthed average entropy, input one of 3 options here.
def information_gain(data, attribute_name, criterion_func):
    total_entropy = criterion_func(data)
    #split the data based on attribute
    splits = split_data(data, attribute_name)
    total_size = len(data)
    weighted_entropy = 0
    for subset_of_data in splits.values():
        subset_size = len(subset_of_data)
        #calculate the weigthed average of the attribute using its subset size * its entropy
        weighted_entropy += (subset_size / total_size) * criterion_func(subset_of_data)
    return total_entropy - weighted_entropy

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


def lets_build_a_tree_baby(data, attributes, criterion_func, depth=0, max_depth=None):
    labels = data.iloc[:, -1]
    
    # Base cases: all labels are the same, or no attributes left, or max depth reached
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    if not attributes or (max_depth is not None and depth == max_depth):
        return get_most_common_label(labels)
    
    # Choose the best attribute to split on
    best_attr = choose_best_attribute(data, attributes, criterion_func)
    
    # Split the dataset on the best attribute
    tree = {best_attr: {}}
    splits = split_data(data, best_attr)
    remaining_attributes = [attr for attr in attributes if attr != best_attr]
    
    # Recursively build the tree for each split
    for attr_value, subset in splits.items():
        tree[best_attr][attr_value] = lets_build_a_tree_baby(subset, remaining_attributes, criterion_func, depth + 1, max_depth)
    return tree

#broken. please help. update: works if it is a pandas dataframe. We in business.
def get_most_common_label(labels):
    labels = label_counts(pd.DataFrame(labels))
    return max(labels, key=labels.get)


def predict(tree, example):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = example[attr]
    subtree = tree[attr].get(value)
    if subtree is None:
        return None 
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

### Part B
def prediction_error(tree, data):
    acc = accuracy(tree, data)
    return 1 - acc

#
def run_different_depths(train_data, test_data, attributes, depths):
    results = {
        'Depth': [],
        'Criterion': [],
        'Train Error': [],
        'Test Error': []
    }
    
    print("Problem 2b:")
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

        # Record results for Information Gain
        results['Depth'].append(depth)
        results['Criterion'].append('Information Gain(Entropy)')
        results['Train Error'].append(train_error_ig)
        results['Test Error'].append(test_error_ig)

        # Record results for Majority Error
        results['Depth'].append(depth)
        results['Criterion'].append('Majority Error')
        results['Train Error'].append(train_error_me)
        results['Test Error'].append(test_error_me)

        
        results['Depth'].append(depth)
        results['Criterion'].append('Gini Index')
        results['Train Error'].append(train_error_gi)
        results['Test Error'].append(test_error_gi)

    return pd.DataFrame(results)

#note: csv files don't have a header. Will have to manually create the column names
train_data = pd.read_csv('car/train.csv', header=None)
test_data = pd.read_csv('car/test.csv', header=None)

train_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

# get the names of the dataset columns, or the attributes
attributes = list(train_data.columns[:-1])

depths = range(1, 7)  
results = run_different_depths(train_data, test_data, attributes, depths)


print("\nPrediction Errors Table:")
print(results)

# 2c. As the depth of the tree gets deeper, both the training error and test error decrease.
print("\n")
print("Problem 2c: As the depth of the tree gets deeper, both the training error and test error generally decrease across all criterion.")