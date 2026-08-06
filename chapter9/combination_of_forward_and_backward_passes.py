# Chapter 9: puts the full forward + backward pass together for a single
# Dense -> ReLU layer, computing dinputs/dweights/dbiases and applying one
# gradient descent update. Then generalizes the same logic into reusable
# Layer_Dense and Activation_ReLU classes (forward + backward methods) that
# later chapters build on to stack multiple layers.
import numpy as np


dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])


inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])


weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([[2, 3, 0.5]])

# Forward pass: Dense layer followed by ReLU activation.
layer_outputs = np.dot(inputs, weights) + biases

relu_outputs = np.maximum( 0 , layer_outputs)

# Backward pass through ReLU: zero out gradients where the pre-activation
# output was <= 0.
drelu = relu_outputs.copy()

drelu[layer_outputs <= 0] = 0

# Backward pass through the Dense layer: gradients w.r.t. inputs, weights,
# and biases, each via the chain rule.
dinputs = np.dot(drelu, weights.T)

dweights = np.dot(inputs.T, drelu)

dbiases = np.sum(drelu, axis=0, keepdims=True)

# One gradient descent step (small fixed learning rate).
weights += -0.001 * dweights

biases += -0.001 * dbiases

print(weights)

print('Biases:')

print(biases)

# Reusable Dense layer: random small initial weights, zero-initialized
# biases, plus forward/backward methods mirroring the manual math above.
class Layer_Dense:

    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))


    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)


# Reusable ReLU activation: passes gradients through unchanged where the
# input was positive, zeroes them out elsewhere.
class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0
        
        
        
# Cross-Entopy Loss derivative


