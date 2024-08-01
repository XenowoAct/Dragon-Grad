"""
Each layer should handle forward prop and back prop.
"""

import numpy as np
from dragongrad.tensor import Tensor
from typing import Dict, Callable
class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
    def forward(self,inputs: Tensor) -> Tensor:
        """
        Propagate data forwards.
        """
        raise NotImplementedError
    
    def backward(self,grad: Tensor) -> Tensor:
        """
        Propagate data backwards.
        """
        raise NotImplementedError
    

class Linear(Layer):
    """
    inputs @ w + b
    """
    def __init__(self,input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self,inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        grad = grad @ w.T
        """
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    Applies an activation function element-wise to the input.
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.f_prime(self.inputs)


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    return 1 - np.tanh(x) ** 2


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)