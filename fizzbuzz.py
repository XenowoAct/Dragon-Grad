from typing import List

import numpy as np

from dragongrad.train import train
from dragongrad.neuralnet import SeqNN
from dragongrad.layers import Linear, Tanh
from dragongrad.optimizers import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x%15 == 0:
        return [0, 0, 0, 1]
    elif x%5 == 0:
        return [0, 0, 1, 0]
    elif x%3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    binary = [x >> i & 1 for i in range(10)]
    return binary


inputs = np.array([binary_encode(x) for x in range(101, 1024)])
targets = np.array([fizz_buzz_encode(x) for x in range(101, 1024)])

net = SeqNN([
    Linear(10, 75),
    Tanh(),
    Linear(75, 4),
])

train(net, inputs, targets, num_epochs=5000,optimizer=SGD(learning_rate=0.001))

for x in range(1,1010):
    predicted = net.forward(binary_encode(x))
    predicted_index = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_index], labels[actual_idx])
    