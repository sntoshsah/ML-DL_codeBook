import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_iters =50, random_state = 42):
        self.lr =lr
        self.n_iters = n_iters
        self.random_state = random_state

    def fit(self, X, y):
        random_gen = np.random.RandomState(self.random_state)
        self.weights = random_gen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.bias = np.float(0.0)
        self.errors = []

        for _ in range(self.n_iters):
            errors = 0
            for xi, target in zip(X,y):
                update = self.lr * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self
    
    def net_input(self, X):
        # Calculate the net input
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """ Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
