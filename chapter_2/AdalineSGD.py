import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


class AdalineSGD:
    def __init__(self, lr = 0.01, n_iter = 10, shuffle_arg = True, random_state = None):
        self.lr = lr
        self.n_iter = n_iter
        self.weight_initialized = False
        self.shuffle_arg = shuffle_arg
        self.random_state = random_state

    def fit(self, X,y):
        self.initialize_weights(X.shape[1])
        self.losses = []
        for i in range(self.n_iter):
            if self.shuffle_arg:
                X, y = self.shuffle(X,y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self.update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses.append(avg_loss)
        return self
    
    def partial_fit(self, X, y):
        if not self.weight_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self. update_weights(xi, target)
        else:
            self.update_weigths(X, y)
        return self
    
    def shuffle(self, X,y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def initialize_weights(self, m):
        r = self.rgen = np.random.RandomState(self.random_state)
        self.weights = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.bias = np.float_(0.0)
        self.weight_initialized = True

    def update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.weights += self.lr * 2.0 * xi * (error)
        self.bias += self.lr * 2.0 * error
        loss = error**2
        return loss
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5 ,1, 0)
    
def dataPrepare():

    data_src = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    dataframe = pd.read_csv(data_src, header=None, encoding='utf-8')
    y = dataframe.iloc[:100,4].values # Select setosa and Versicolor data
    y = np.where(y == "Iris-setosa", 0,1)
    X = dataframe.iloc[:100, [0,2]].values # Extract sepal length and petal length
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()    # Standardarization : (x-mean)/std
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()    # Standardarization : (x-mean)/std
    return X_std, y

def plot_decision_regions(X,y, classifier, resolution = 0.02):
    # Setup marker generator and color map
    markers = ('o', 's', '^','<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() -1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha = 0.3, cmap= cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class Examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl, 0],
                    y = X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    label = f"class {cl}",
                    edgecolors='black')
        

X_std, y = dataPrepare()

if __name__ == "__main__":
    ada_sgd = AdalineSGD(n_iter =  15, lr=0.01, random_state = 42)
    ada_sgd.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plt.title(" Adaline- Stochastic Gradient Descent")
    plt.xlabel("Sepal Length [Standarized]")
    plt.ylabel("Petal Length [Standarized]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.plot(range(1, len(ada_sgd.losses) + 1),ada_sgd.losses, marker = 'o')
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.tight_layout()
    plt.show()

