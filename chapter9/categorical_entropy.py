# Chapter 9: backward pass (gradient) for Categorical Cross-Entropy loss.
# Reuses the Loss base class from chapter 5 (cross_entropy_loss.py) and adds
# a `backward` method so this loss can participate in backpropagation.
import sys
from pathlib import Path

import numpy as np

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