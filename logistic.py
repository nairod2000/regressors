import numpy as np

# TODO find the derivitaive of the weight and bias and implement it

class LogisticRegressor:

    def __init__(self):
        self.lr = 0.05
        self.i = 1000
        self.weight = None
        self.bias = None
        self.n_examples = None

    def fit(self, X, y):
        self.n_examples = len(X)

        for _ in range(self.i):
            self.update(X, y)

    def update(self, X, y):
        self.weight -= self.lr * self.weight_derivative()
        self.bias -= self.lr * self.bias_derivative()

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * (self.weight * x + self.bias)))

    def weight_derivative(self, X, y):
        error = y - (self.weight * X + self.bias)
        delta = np.sum(error * X) / self.n_examples
        return delta

    def bias_derivative(self):
        error = y - (self.weight * X + self.bias)
        delta = np.sum(error) / self.n_examples
        return delta

    def predict(self, x):
        return self._sigmoid(x)
