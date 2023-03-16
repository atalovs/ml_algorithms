import numpy as np


class LinearRegression:
    """Linear Regression

    Parameters:
    ----------
    fit_intercept: bool
        Whether to calculate the intercept for this model.
        If set to False, no intercept will be used in calculations
    gradient_descent: bool
        Whether to use gradient descent to find the coefficients.
        If set to False, normal equation will be used
    n_iter: int
        Number of iterations if gradient descent selected
        Will be ignored if gradient_descent parameter set to False
    learning_rate: float
        Learning rate for gradient descent
        Will be ignored if gradient_descent parameter set to False
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        gradient_descent: bool = True,
        n_iter: int = 500,
        learning_rate: float = 0.001,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.gradient_descent = gradient_descent
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model

        Parameters:
        ----------
        X: np.ndarray
            Features
        y: np.ndarray
            Target values

        Returns
        ----------
        self : LinearRegression
            Fitted Linear Regression
        """
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        if self.gradient_descent:
            self.coef_ = np.random.rand(
                X.shape[1],
            )
            for _ in range(self.n_iter):
                y_pred = X @ self.coef_
                grad = np.mean(-2 * (y - y_pred) @ X)
                self.coef_ -= self.learning_rate * grad
        else:
            self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make a predictions for a given X (features)

        Parameter:
        ----------
        X: np.ndarray
            Features to make a prediction

        Returns:
        ----------
        y_pred: np.ndarray
            Predictions
        """
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        assert X.shape[1] == self.coef_.shape[0], "X has different number of features!"
        y_pred = X @ self.coef_
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared

        Parameters:
        ----------
        X: np.ndarray
            Features
        y: np.ndarray
            Target values

        Returns:
        ----------
        score: float
            R-squared
        """
        y_pred = self.predict(X)
        # Residual sum of squares
        rss = np.sum((y - y_pred) ** 2)
        # Total sum of squares
        tss = np.sum((y - np.mean(y)) ** 2)
        r_2 = 1 - rss / tss
        return r_2
