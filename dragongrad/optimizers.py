"""
An optimizer serves to update the parameters of a neural network based on the results of back prop.
"""

from dragongrad.neuralnet import SeqNN

class Optimizer:
    def step(self, net: SeqNN) -> None:
        raise NotImplementedError
    

class SGD(Optimizer):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def step(self, net: SeqNN) -> None:
        for param, grad in net.params_and_grads():
            param -= self.learning_rate * grad