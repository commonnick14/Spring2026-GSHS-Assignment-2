import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    # Convert to float
    X = x_train.astype(np.float64)
    y = y_train.astype(np.float64).reshape(-1)
    Xt = x_test.astype(np.float64)

    # ===== 핵심: feature standardization =====
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    Xt = (Xt - mean) / std

    n, d = X.shape

    # Add bias
    Xb = np.hstack([np.ones((n, 1)), X])
    Xtb = np.hstack([np.ones((Xt.shape[0], 1)), Xt])

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # Initialize weights
    w = np.zeros(d + 1)

    # Gradient Descent (이제 안정적으로 수렴함)
    lr = 0.1
    iters = 3000

    for _ in range(iters):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        w -= lr * grad

    probs = sigmoid(Xtb @ w)
    return (probs >= 0.5).astype(int)
