# A layer of 3 neurons (each with its own weights/bias) computed manually
# with plain Python loops, before switching to numpy for the same operation.
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

weights = [weights1, weights2, weights3]
biases = [2, 3, 0.5]

# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# outputs = [ inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + inputs[4] * weights1[4] + bias1,
#             inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + inputs[4] * weights2[4] + bias2,
#             inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + inputs[4] * weights3[4] + bias3 ]

# for i in range(len(inputs)):
#     bias1 = 2
#     bias2 = 3
#     bias3 = 0.5
    
#     output1 = inputs[i] * weights1[i]
#     output2 = inputs[i] * weights2[i]
#     output3 = inputs[i] * weights3[i]
    
# outputs = [output1 + bias1, output2 + bias2, output3 + bias3]

layer_outputs = []

# Outer loop: one pass per neuron (its own weight vector + bias).
# Inner loop: dot-product that neuron's weights against the shared inputs.
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        # print(f"n_input: {n_input}, weight: {weight}")
        neuron_output += n_input * weight
        # print(f"neuron_output: {neuron_output}")

    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)

# print(outputs)

print(layer_outputs)


