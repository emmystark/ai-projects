# Chapter 4: building the Softmax activation step by step. Softmax turns
# raw layer outputs into a probability distribution: exponentiate each
# value (so everything is positive and larger values dominate), then
# normalize by the sum so all outputs add up to 1.
layer_outputs = [4.8, 1.21, 2.385]

E = 2.71828182846

exp_values = []

for output in layer_outputs:
    exp_values.append(E ** output)

print(f"Exponential values: {exp_values}")


norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(f"Normalized values: {norm_values}")


# with numpy

import numpy as np

exp_values = np.exp(layer_outputs)

# probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)



layer_outputs1 = np.array([[4.8, 1.21, 2.385],
                           [8.9, -1.81, 0.2],
                           [1.41, 1.051, 0.026]])

print(f"Sum withiout axis: {np.sum(layer_outputs1)}")

print(f"Sum with axis=0: {np.sum(layer_outputs1, axis=0)}")

# axis=1 sums each row (each sample) independently — this is what Softmax
# needs, since normalization must happen per-sample, not across the batch.
for i in layer_outputs1:
    print(sum(i))

print("So we can sum axis 1, but note the current shape: ")
print(np.sum(layer_outputs1, axis=1, keepdims=True))