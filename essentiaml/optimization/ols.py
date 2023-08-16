import numpy as np


class OLS:
    """Ordinary Least Squares (OLS) regression model."""
    def __init__(self):
        """Initialize the OLS model."""
        self.coef_ = None
        self.intercept_ = None

    def add_constant(self, X):
        """Add a column of ones to the left side of the matrix X."""
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X, y):
        """
        Fit the OLS model to the given data.
        Parameters:
            X (ndarray): The feature matrix of shape (n_samples, n_features).
            y (ndarray): The target vector of shape (n_samples,).
        """
        self._validate_inputs(X, y, fit=True)

        padded_X = self.add_constant(X)
        self.coef_ = np.linalg.inv(padded_X.T @ padded_X) @ padded_X.T @ y
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        """
        Predict the target variable for a given set of features.
        Parameters:
            X (ndarray): The feature matrix of shape (n_samples, n_features).

        Returns:
            ndarray: The predicted target vector of shape (n_samples,).
        """
        self._validate_inputs(X, None)
        return self.intercept_ + X @ self.coef_

    def _validate_inputs(self, X, y=None, fit=False):
        """Validate the inputs to the model."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if y is not None:
            if y.ndim != 1:
                raise ValueError("y must be a one-dimensional array.")
            if X.shape[0] != y.shape[0]:
                raise ValueError("The number of samples in X and y must match.")

        if self.coef_ is not None and X.shape[1] != self.coef_.shape[0]:
            raise ValueError("The number of features in X must match the number of coefficients.")

        if fit:
            padded_X = self.add_constant(X)
            if np.linalg.matrix_rank(padded_X) < padded_X.shape[1]:
                raise ValueError("The matrix X is singular and cannot be inverted. Check for multicollinearity or "
                                 "other issues.")
