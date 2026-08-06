# Chapter 9: backpropagation through a Dense layer's weights and biases
# (batch of 3 samples). dbiases sums gradients across the batch (axis=0)
# since each sample contributed independently to the bias's effect.
# dweights uses inputs.T @ dvalues to get the gradient w.r.t. each weight.
import numpy as np


dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

biases = np.array([[2, 3, 0.5]])


dbiases = np.sum(dvalues, axis=0, keepdims=True)

print(dbiases)

# weights = np.array([[0.2, 0.8, -0.5, 1],
#                     [0.5, -0.91, 0.26, -0.5],
#                     [-0.26, -0.27, 0.17, 0.87]]).T

inputs = np.array([[1, 2, 3, 2.5],
                    [2., 5., -1, 2],
                    [-1.5, 2.7, 3.3, -0.8]])


# dx0 = sum(weights[0])*dvalues[0]
# dx1 = sum(weights[1])*dvalues[0]
# dx2 = sum(weights[2])*dvalues[0]
# dx3 = sum(weights[3])*dvalues[0]


# dinputs = np.array([dx0, dx1, dx2, dx3])
# dinputs = np.dot(dvalues.T, weights)
dweights = np.dot(inputs.T, dvalues)

print(dweights)