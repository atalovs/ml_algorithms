import numpy as np

class KNeighborsClassifier:
    """K Nearest Neighbors Classifier

    Parameters:
    ---
    n_neighbors: int
        The number of closest neighbors to determine a class
    """
    def __init__(self, n_neighbors: int = 3) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model

        Parameters:
        ----------
        X: np.ndarray
            Features
        y: np.ndarray
            Target values

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier
        """
        self.X = X
        self.y = y
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for given X

        Parameter:
        ----------
        X_new: np.ndarray
            Features to make a prediction

        Returns:
        ----------
        y_pred: np.ndarray
            Predicted class labels
        """
        assert self.X.shape[1] == X_new.shape[1], "X has different number of features!"
        y_pred = np.zeros(X_new.shape[0])
        for i, sample in enumerate(X_new):
            # Compute distances between one point and all samples in X: training set
            # and sort by distances (the indices)
            distances = np.argsort([self._distance(sample, p) for p in self.X])
            # Select first k neighbors
            k_nearest = distances[: self.n_neighbors]
            # Make a prediction as the most common label of the first k neighbors
            y_pred[i] = self._most_common_label(self.y[k_nearest])

        return y_pred

    def _distance(self, p_1: np.ndarray, p_2: np.ndarray) -> float:
        """Helper function to calculate Euclidean distance between two points

        Parameters:
        ----------
        p_1: np.ndarray
            Coordinates of the first point
        p_2: np.ndarray
            Coordinates of the second point

        Returns:
        ----------
        distance: float
            Euclidean distance between two points
        """
        distance = np.sqrt(np.sum((p_1 - p_2) ** 2))
        return distance

    def _most_common_label(self, y: np.ndarray) -> int:
        """Helper function to find most common label

        Parameters:
        ----------
        y: np.ndarray
            Class labels
        
        Returns:
        ----------
        label: int
            Most common label
        """
        return np.argmax(np.bincount(y))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculates accuracy of the predictions with given X and y

        Parameters:
        ----------
        X: np.ndarray
            Features
        y: np.ndarray
            Target values

        Returns:
        ----------
        score: float
            Classification accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
