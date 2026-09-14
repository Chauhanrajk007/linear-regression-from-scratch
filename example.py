import numpy as np
from linear_regression import LinearRegression

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X, y)

print(f"\nFinal weight (slope): {model.w:.4f}")
print(f"Final bias (intercept): {model.b:.4f}")
print("Predictions:", np.round(model.predict(X), 2))