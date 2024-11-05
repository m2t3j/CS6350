import numpy as np
import pandas as pd



class Perceptron:

    def __init__(self,learning_rate = .01, n_epochs= 10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None 
        self.epoch_errors = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        #create true ground truth labels using the step activation function
        y_true = np.where(y <= 0, -1, 1)
        
        #iterate through and calculate the weights and bias
        for epoch in range(self.n_epochs):
            #keep track of indexs and the value of a feature
            for index, x_i in enumerate(X):
                #linear model output w^tx
                output = np.dot(x_i,self.weights) + self.bias
                #predicted y values are based on the step activation function. If y is greater than 0, then 1, otherwise -1
                y_pred = np.where(output >= 0, 1, -1)


                #weight updates
                if y_pred != y_true[index]:
                    self.weights += self.learning_rate * (y_true[index] - y_pred) * x_i
                    self.bias += self.learning_rate * (y_true[index] - y_pred)
                

            error = self.avg_prediction_error(X, y)
            self.epoch_errors.append(error)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Error: {error}")


    def predict(self, X):
        y_predictions = np.dot(X, self.weights) + self.bias
        return np.where(y_predictions >= 0, 1, -1)
    
    def avg_prediction_error(self,X,y):
        preds = self.predict(X)
        y = np.where(y >=0, 1, -1)
        accuracy = np.mean(preds == y)
        return 1 - accuracy
    



if __name__ == "__main__":

    train = pd.read_csv("Perceptron/data/bank-note/train.csv")
    test = pd.read_csv("Perceptron/data/bank-note/test.csv")

    X_train = train.iloc[:, :-1].values 
    y_train = train.iloc[:, -1].values

    #print(y_train)

    p = Perceptron()

    p.fit(X_train, y_train)
    
    print("Perceptron Weights: ", p.weights)
    print("Perceptron Bias: ", p.bias)

    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    avg_train_error = np.array(p.epoch_errors).mean()
    print("Average Train Error: ", avg_train_error)

    test_error = p.avg_prediction_error(X_test, y_test) 
    print("Average Test Error:", test_error)
    print("\n")
    


        


