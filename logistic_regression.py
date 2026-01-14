import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    '''
    Implements the logistic regression algorithm.

    Parameters:
        - x_train: Training features of shape (n_samples, 2).
        - y_train: Training labels (0/1)
        - x_test: Test features of shape (n_samples, 2).

    Returns:
        y_pred: Predicted labels for the test set
    '''

    # Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Predict labels for test data
    y_pred = model.predict(x_test)

    return y_pred
