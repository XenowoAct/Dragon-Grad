"""
A loss function measures how accurate the predictions are. This serves as guidance when updaing parameters.
"""

from dragongrad.tensor import Tensor
import numpy as np

class Loss:
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class TSE(Loss):
    """
    Total Squared Error;
    sum((predicted - actual)**2)
    """
    def loss(self,predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
