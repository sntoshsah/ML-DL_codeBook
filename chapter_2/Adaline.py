import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adaline -> Adaptive Linear Neuron

class AdalineGD:
    def __init__(self, lr = 0.01, n_iter = 50, random_state = 42):
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.weight = rgen.normal(loc=0.0, scale=0.01, size = X.shape[1])
        self.bias = np.float_(0.)
        self.losses = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.weight += self.lr * 2.0 * X.T.dot(errors)/X.shape[0]
            self.bias += self.lr * 2.0 * errors.mean()
            loss = (errors **2).mean()
            self.losses.append(loss)

        return self
    
    def net_input(self, X):
        return np.dot(X, self.weight) + self.bias
    
    def activation(self, X):
        return X       # linear Activation
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    

# Plot the loss against the number of epochs for the two different learning rate (lr )
if __name__ == "__main__":

    data_src = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    dataframe = pd.read_csv(data_src, header=None, encoding='utf-8')
    y = dataframe.iloc[:100,4].values # Select setosa and Versicolor data
    y = np.where(y == "Iris-setosa", 0,1)

    X = dataframe.iloc[:100, [0,2]].values # Extract sepal length and petal length




    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,4))
    ada1 = AdalineGD(n_iter=50, lr=0.1).fit(X, y)
    ax[0].plot(range(1, len(ada1.losses) +1), np.log10(ada1.losses), marker = 'o')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("log(Mean Squared Error)")
    ax[0].set_title('Adaline - Learning Rate 0.1')

    ada2 = AdalineGD(n_iter=50, lr = 0.0001).fit(X,y)
    ax[1].plot(range(1, len(ada2.losses) +1), np.log10(ada2.losses), marker = 'o')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("log(Mean Squared Error)")
    ax[1].set_title('Adaline - Learning Rate 0.0001')
    plt.show()