# Chapter 9: vectorized ReLU backward pass over a batch of samples using numpy.
# z holds the pre-activation values (layer output before ReLU); dvalues is the
# gradient flowing in from the next layer. ReLU's derivative is 1 where z > 0
# and 0 otherwise, so we zero out gradients wherever z was <= 0.
import numpy as np

z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])



dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# drelu = np.zeros_like(z)
drelu = dvalues.copy()

# drelu[z > 0] = 1
drelu[z <= 0] = 0

print(drelu)

# drelu *= dvalues

# print(drelu)

