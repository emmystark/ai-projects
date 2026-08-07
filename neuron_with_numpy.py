import numpy as np

# Single neuron with numpy
# np.dot replaces manually summing inputs[i] * weights[i] for each i.

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = np.dot(inputs, weights) + bias
print(output)

# Layered neurons with numpy
# np.dot(layer_weights, inputs) computes all 3 neurons' outputs in one call
# since layer_weights is a matrix (one row per neuron).


layer_weights = [[0.2, 0.8, -0.5, 1.0],
                 [0.5, -0.91, 0.26, -0.5],
                 [-0.26, -0.27, 0.17, 0.87]]
layer_biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(layer_weights, inputs) + layer_biases
print(layer_outputs)


# Batch training with numpy
# A batch groups multiple samples together so the whole network can process
# them in one vectorized pass instead of one sample at a time.

batch = [[1.0, 2.0, 3.0, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8],
         [1.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0],
         [1.0, 2.0, 3.0, 4.0],
         [0.5, 1.5, 2.5, 3.5]]