import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n = len(X)
        self.w = 0
        self.b = 0

        for _ in range(self.epochs):
            y_pred = self.w * X + self.b

            dw = (2 / n) * np.sum(X * (y_pred - y))
            db = (2 / n) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if _ % 200 == 0:
                mse = (1 / n) * np.sum((y_pred - y) ** 2)
                print(f"Epoch {_:3d} | w={self.w:.4f} b={self.b:.4f} | MSE={mse:.4f}")

    def predict(self, X):
        return self.w * X + self.b