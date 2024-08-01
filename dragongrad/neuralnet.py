"""
A sequential NN consists of a list of layers where data flows sequentially.
"""
from typing import Sequence, Iterator, Tuple
from dragongrad.tensor import Tensor
from dragongrad.layers import Layer

class SeqNN:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: Tensor) -> Tensor:
        x = grad
        for layer in reversed(self.layers):
            x = layer.backward(x)
        return x
    
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad