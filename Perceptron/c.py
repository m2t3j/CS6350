import numpy as np
import pandas as pd

class AveragePerceptron:
    def __init__(self, learning_rate=0.01, n_epochs=10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None 
        self.bias = 0
        self.weight_sums = None  # To keep track of the cumulative weights

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.weight_sums = np.zeros(num_features)  # Initialize weight sums
        self.bias = 0
        total_updates = 0  # To keep track of the number of updates for averaging

        # Create true ground truth labels using the step activation function
        y_true = np.where(y <= 0, -1, 1)

        for epoch in range(self.n_epochs):
            for index, x_i in enumerate(X):
                output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if output >= 0 else -1
                
                # Update weights if prediction is incorrect
                if y_pred != y_true[index]:
                    # Update the weights and bias
                    self.weights += self.learning_rate * (y_true[index] - y_pred) * x_i
                    self.bias += self.learning_rate * (y_true[index] - y_pred)

                    # Accumulate weights for averaging
                    self.weight_sums += self.weights
                    total_updates += 1  # Increment the number of updates

        # Average the weights by the number of updates
        if total_updates > 0:
            self.weights = self.weight_sums / total_updates

    def predict(self, X):
        y_predictions = np.dot(X, self.weights) + self.bias
        return np.where(y_predictions >= 0, 1, -1)

    def avg_prediction_error(self, X, y):
        preds = self.predict(X)
        y_true = np.where(y <= 0, -1, 1)  # Ensure consistent label transformation
        accuracy = np.mean(preds == y_true)
        return 1 - accuracy  # Return error rate


if __name__ == "__main__":
    # Load training and testing data
    train = pd.read_csv("Perceptron/data/bank-note/train.csv")
    test = pd.read_csv("Perceptron/data/bank-note/test.csv")

    X_train = train.iloc[:, :-1].values 
    y_train = train.iloc[:, -1].values

    # Initialize and fit the Average Perceptron model
    ap = AveragePerceptron()
    ap.fit(X_train, y_train)

    # Report the learned weight vector
    print("Learned Weight Vector:", ap.weights)

    # Evaluate on test data
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    test_error = ap.avg_prediction_error(X_test, y_test)
    print("Average Test Error:", test_error)


    print("Compared to the Voted Perceptron model in part b, the weights are much closer to 0 and the test error was slighly larger, but the two models generally had the same performance. ")
    print("\n")
