import numpy as np
from libs.models.logistic_regression import LogisticRegression

class LogisticRegressionPenalized(LogisticRegression):
    def __init__(self, num_features: int, lambda_: float = 0.1):
        super().__init__(num_features)
        self.lambda_ = lambda_
    
    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """
        Function to update the weights in-place with L2 regularization.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        # Apply the L2 regularization term to the gradient
        regularization_term = self.lambda_ * self.parameters
        regularization_term[0] = 0  # Do not regularize the bias term
        
        # Update the parameters with the gradient and regularization
        self.parameters += lr * (gradient - regularization_term)