import matplotlib.pyplot as plt

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter3.dense_layer import Layer_Dense

from chapter4.ReLu_class import Activation_ReLu
from chapter4.softmax_class import  Activation_Softmax


from chapter5.cross_entropy_loss import Loss_CategoricalCrossentropy

import nnfs

from nnfs.datasets import vertical_data

nnfs.init()

# Chapter 6: "random search" optimization — before gradient-based training
# (chapters 9-10), this brute-forces better weights by randomly nudging
# them each iteration and only keeping the change if it lowered the loss.
# It's a naive baseline meant to show why this approach doesn't scale, in
# contrast to the gradient descent covered later.
X, y =vertical_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLu()
dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# Track the best-known parameters found so far, and the lowest loss they
# achieved, so we can always roll back a bad random nudge.
lowest_loss = 99999999

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')

# plt.show()


for iteration in range(10000):

    # Randomly nudge every weight/bias a small amount (no gradient info
    # involved — this is just blind trial and error).
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    # Note: this uses `=` instead of `+=` like the other three lines, so
    # dense2.weights gets replaced with a fresh random matrix each
    # iteration rather than nudged from its current value.
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Forward pass with the nudged parameters to see how they perform.
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)


    loss = loss_function.calculate(activation2.output, y)


    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    # Keep the nudge only if it improved the loss; otherwise revert to
    # the best parameters found so far and try a different random nudge
    # next iteration.
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration, 'loss:', loss, 'acc:', accuracy)

        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()

    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        lowest_loss = loss

        # print(lowest_loss)