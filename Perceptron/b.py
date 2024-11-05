import numpy as np
import pandas as pd

class VotedPerceptron:
    def __init__(self, learning_rate=0.01, n_epochs=10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = []  
        self.biases = []   
        self.counts = []   

    def fit(self, X, y):
        num_samples, num_features = X.shape
        weights = np.zeros(num_features)
        bias = 0

        # Create true ground truth labels using the step activation function
        y_true = np.where(y <= 0, -1, 1)

        for epoch in range(self.n_epochs):
            for index, x_i in enumerate(X):
                output = np.dot(x_i, weights) + bias
                y_pred = 1 if output >= 0 else -1
                
                # Update weights if prediction is incorrect
                if y_pred != y_true[index]:
                    # Update the weights and bias
                    weights += self.learning_rate * (y_true[index] - y_pred) * x_i
                    bias += self.learning_rate * (y_true[index] - y_pred)

                    # Store the current weight vector and bias, and append 1 to counts until the prediction is correct
                    self.weights.append(weights.copy())
                    self.biases.append(bias)
                    self.counts.append(1) 
                else:
                    # If the prediction is correct, just increment the count of the last weight vector
                    if self.weights:
                        self.counts[-1] += 1

        # Remove duplicates and keep track of their counts
        self.weights, self.biases, self.counts = self._aggregate_weights(self.weights, self.biases, self.counts)

    def _aggregate_weights(self, weights, biases, counts):
        unique_weights = []
        unique_biases = []
        unique_counts = []
        for w, b, c in zip(weights, biases, counts):
            if len(unique_weights) == 0 or not np.array_equal(w, unique_weights[-1]):
                unique_weights.append(w)
                unique_biases.append(b)
                unique_counts.append(c)
            else:
                unique_counts[-1] += c
        return unique_weights, unique_biases, unique_counts

    def predict(self, X):
        num_samples = X.shape[0]
        total_votes = np.zeros(num_samples)

        for w, b, c in zip(self.weights, self.biases, self.counts):
            predictions = np.dot(X, w) + b  
            total_votes += np.where(predictions >= 0, c, -c) 

        return np.where(total_votes >= 0, 1, -1)

    def avg_prediction_error(self, X, y):
        preds = self.predict(X)
        y_true = np.where(y <= 0, -1, 1) 
        accuracy = np.mean(preds == y_true)
        return 1 - accuracy 

if __name__ == "__main__":
    # Load training and testing data
    train = pd.read_csv("Perceptron/data/bank-note/train.csv")
    test = pd.read_csv("Perceptron/data/bank-note/test.csv")

    X_train = train.iloc[:, :-1].values 
    y_train = train.iloc[:, -1].values

    # Initialize and fit the Voted Perceptron model
    vp = VotedPerceptron()
    vp.fit(X_train, y_train)

    # Report the list of distinct weight vectors and their counts
    print("Distinct Weight Vectors and their Counts:")
    for w, b, c in zip(vp.weights, vp.biases, vp.counts):
        print(f"Weight Vector: {w}, Bias: {b}, Count: {c}")

    # Evaluate on test data
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    test_error = vp.avg_prediction_error(X_test, y_test)
    print("Average Test Error:", test_error)
    print("\n")
