import pandas as pd
import math

def load_data(filename):
   return pd.read_csv(filename)

def label_counts(data):
   #label_counts = data.iloc[:-1].valuecounts()
   label_counts = {}
   #look at last column
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

### gini index = 1 - (p+^2 + p-^2)
def gini_index(data):
   label_count = label_counts(data)
   total_num_lables = len(data)
   gini = 1 - sum((count / total_num_lables) ** 2 for count in label_count.values())
   return gini

def majority_error(data):
   label_count = label_counts(data)
   total_num_labels = len(data)
   majority_label = max(label_count.values())
   return 1 - (majority_label / total_num_labels)
