# Chapter 9: backward pass (gradient) for Categorical Cross-Entropy loss.
# Reuses the Loss base class from chapter 5 (cross_entropy_loss.py) and adds
# a `backward` method so this loss can participate in backpropagation.
import sys
from pathlib import Path

import numpy as np
import nnfs


nnfs.init()

# Make the project root importable so `from chapter5...` resolves regardless
# of which directory this script is run from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter5.cross_entropy_loss import Loss


# np.eye(5) makes a 5x5 identity matrix; indexing a row out of it gives a
# one-hot vector. This demonstrates how a sparse class label (e.g. "1" or
# "4") gets converted into a one-hot vector for use in the gradient formula
# below (y_true needs to be one-hot to line up with dvalues elementwise).
one_hot_coded = np.eye(5)[1]
print(one_hot_coded)

one_hot_coded1 = np.eye(5)[4]
print(one_hot_coded1)



softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])


class_targets = np.array([0, 1, 1])




# Categorical Cross-Entropy loss with a backward pass, extending the
# forward-only version from chapter 5.
class Loss_CategoricalCrossentropy(Loss):

    def backward(self, dvalues, y_true):
        # Number of samples in the batch, and number of classes (labels)
        # per sample, inferred from the shape of dvalues (predictions).
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If labels are sparse (e.g. [0, 1, 1]) rather than one-hot,
        # convert them to one-hot so they line up elementwise with dvalues.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Gradient of Categorical Cross-Entropy w.r.t. predicted
        # probabilities: -y_true / y_pred, derived from
        # d/dy_pred[-sum(y_true * log(y_pred))].
        self.dinputs = -y_true / dvalues

        # Normalize by number of samples so the gradient's scale doesn't
        # grow with batch size (keeps optimizer steps batch-size independent).
        self.dinputs = self.dinputs / samples
        
        
    


# Begin of Softmax

softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1, 1)

print(softmax_output)


print(np.eye(softmax_output.shape[0]))


# print(softmax_output * np.eye(softmax_output.shape[0]))

print(np.diagflat(softmax_output))


print(np.dot(softmax_output, softmax_output.T))


print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))


# Softmax activation, with the backward pass generalizing the manual
# steps above into a reusable method that works sample-wise over a batch.
class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                          single_dvalues)


class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        
        self.dinputs = self.dinputs / samples    
        
        
        
        
        

softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

activation = Activation_Softmax()

activation.output = softmax_outputs

loss = Loss_CategoricalCrossentropy()

loss.backward(softmax_outputs, class_targets)

activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print('Gradients: combined loss and activation:')

print(dvalues1)
print('Gradients: seperate loss and activation:')

print(dvalues2)


# Compare performance of the combined backward pass
# (Activation_Softmax_Loss_CategoricalCrossentropy) against the separate
# Activation_Softmax + Loss_CategoricalCrossentropy backward passes, since
# combining them algebraically (dvalues - 1, divided by samples) skips
# building the Jacobian matrix entirely.
from timeit import timeit


def f1():
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs


def f2():
    activation = Activation_Softmax()
    activation.output = softmax_outputs
    loss = Loss_CategoricalCrossentropy()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs


t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(t2 / t1)


