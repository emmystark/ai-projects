# Chapter 5: computing classification accuracy from Softmax outputs —
# separate from loss, since loss measures "how wrong" while accuracy just
# measures "how often correct."
import numpy as np

softmax_outputs = [[0.7, 0.2, 0.1],
                   [0.5, 0.1, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = np.array([0, 1, 1])

# Predicted class per sample = index of its highest probability.
predictions = np.argmax(softmax_outputs, axis=1)

# If targets are one-hot encoded (2D), convert to class indices (1D)
# so they can be compared directly against `predictions`.
if len(class_targets.shape) == 2:
    class_targets. np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions==class_targets)


print('acc', accuracy)