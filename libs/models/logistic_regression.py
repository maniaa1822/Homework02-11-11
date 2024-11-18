import numpy as np
from libs.math import sigmoid

class LogisticRegression:
    def __init__(self, num_features : int):
        self.parameters = np.random.normal(0, 0.01, num_features)
        
    def predict(self, x: np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: it's the input data matrix.

        Returns:
            preds: the predictions of the input features.
        """
        preds = sigmoid(np.dot(x, self.parameters))
        return preds
    
    @staticmethod
    def likelihood(preds, y: np.array) -> np.array:
        """
        Compute the log-likelihood given predictions and true labels.

        Args:
            preds: Predicted probabilities (from the sigmoid function).
            y: True labels (0 or 1).

        Returns:
            log_l: The log-likelihood of the model given data.
        """
        # Ensure numerical stability by clipping predictions
        preds = np.clip(preds, 1e-10, 1 - 1e-10)

        # Calculate the log-likelihood
        log_l = np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
        return log_l
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        self.parameters += lr * gradient
        
    @staticmethod
    def compute_gradient(x: np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: it's the input data matrix.
            y: the label array.
            preds: the predictions of the input features.

        Returns:
            gradient: the gradient of the log likelihood.
        """
        errors = y - preds
        gradient = np.dot(x.T, errors) / x.shape[0]
        return gradient

