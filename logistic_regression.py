import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    X = x_train.astype(np.float64)
    y = y_train.astype(np.float64).reshape(-1)
    X_test = x_test.astype(np.float64)

    n, d = X.shape
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)
    Xtb = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)

    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    w = np.zeros(d + 1, dtype=np.float64)

    lr = 0.1
    iters = 5000
    l2 = 1e-4

    for _ in range(iters):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / n
        grad[1:] += l2 * w[1:]
        w -= lr * grad

    probs = sigmoid(Xtb @ w)
    return (probs >= 0.5).astype(np.int64)
