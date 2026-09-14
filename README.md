# Linear Regression from Scratch

Built linear regression using gradient descent — no sklearn, no machine learning libraries.

## The Math

**Hypothesis:** `y_pred = w * X + b`

**Cost function (MSE):**

```
J = (1/n) * sum((y_pred - y)^2)
```

**Gradients:**

```
dw = (2/n) * sum(X * (y_pred - y))
db = (2/n) * sum(y_pred - y)
```

**Update rule:**

```
w = w - lr * dw
b = b - lr * db
```

We repeat the update for `epochs` iterations until the cost converges.

## How it works

1. Start with `w = 0`, `b = 0`
2. Predict with current weights
3. Compute the error
4. Take the derivative (gradient) of the cost w.r.t `w` and `b`
5. Move weights in the direction that reduces the error (gradient descent)
6. Repeat

## Usage

```bash
pip install numpy
python example.py
```

## Output

```
Epoch   0 | w=0.0142 b=0.0040 | MSE=16.2576
...
Final weight (slope): 0.5820
Final bias (intercept): 2.2360
Predictions: [3. 3. 3. 3. 3.]
```

## Files

- `linear_regression.py` — the model class
- `example.py` — a runnable example