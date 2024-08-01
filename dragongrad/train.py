"""
a function that can train a neuralnet
"""

from dragongrad.neuralnet import SeqNN
from dragongrad.data import BatchIterator, DataIterator
from dragongrad.loss import Loss, TSE
from dragongrad.optimizers import Optimizer, SGD
from dragongrad.tensor import Tensor

def train(net: SeqNN,
        inputs: Tensor,
        targets: Tensor,
        num_epochs: int = 5000,
        iterator: DataIterator = BatchIterator(),
        loss: Loss = TSE(),
        optimizer: Optimizer = SGD(learning_rate=0.01),
    ) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)