import numpy as np

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """
    Implements logistic regression (binary classification) using numpy only.

    Parameters:
        x_train: (n_samples, 2)
        y_train: (n_samples,) with labels 0/1
        x_test:  (n_test, 2)

    Returns:
        y_pred: (n_test,) predicted labels 0/1
    """
    # Ensure float
    X = x_train.astype(np.float64)
    y = y_train.astype(np.float64).reshape(-1)
    X_test = x_test.astype(np.float64)

    n, d = X.shape  # d should be 2

    # Add bias term
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)          # (n, d+1)
    Xtb = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)  # (m, d+1)

    # Sigmoid (numerically stable)
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # Initialize weights
    w = np.zeros(d + 1, dtype=np.float64)

    # Hyperparameters (safe defaults)
    lr = 0.1
    iters = 5000
    l2 = 1e-4  # small regularization helps stability (bias not regularized)

    for _ in range(iters):
        p = sigmoid(Xb @ w)                 # (n,)
        grad = (Xb.T @ (p - y)) / n         # (d+1,)

        # L2 regularization (do not regularize bias term)
        grad[1:] += l2 * w[1:]

        w -= lr * grad

    # Predict on test
    probs_test = sigmoid(Xtb @ w)
    y_pred = (probs_test >= 0.5).astype(np.int64)

    return y_pred
