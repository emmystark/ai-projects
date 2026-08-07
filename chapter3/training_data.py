# Chapter 3: generating and visualizing the spiral training dataset, and
# showing how a layer's weights/biases are initialized (small random
# weights, zero biases) before any training happens.
import numpy as np

from nnfs.datasets import spiral_data

import matplotlib.pyplot as plt

x, y = spiral_data(samples=100, classes=3)

# plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')
# plt.show()


import nnfs

nnfs.init()

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))

print(weights)
print(biases)

# print(np.random.randn(2, 5))

# print(np.zeros((2, 5)))