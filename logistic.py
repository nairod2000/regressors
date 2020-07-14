import numpy as np

# TODO find the derivitaive of the weight and bias and implement it

class LogisticRegressor:

    def __init__(self):
        self.lr = 0.05
        self.i = 1000
        self.weight = 1
        self.bias = 1
        self.n_examples = None

    def fit(self, X, y):
        self.n_examples = len(X)

        for _ in range(self.i):
            self.update(X, y)

    def update(self, X, y):
        self.weight -= self.lr * self.weight_derivative(X, y)
        self.bias -= self.lr * self.bias_derivative(X, y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * (self.weight * x + self.bias)))

    def weight_derivative(self, X, y):
        error = self._sigmoid(self.weight * X + self.bias) - y
        delta = np.sum(error * X) / self.n_examples
        return delta

    def bias_derivative(self, X, y):
        error = self._sigmoid(self.weight * X + self.bias) - y
        delta = np.sum(error) / self.n_examples
        return delta

    def predict(self, x):
        return self._sigmoid(x)
