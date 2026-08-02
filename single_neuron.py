inputs = [1, 2, 2.5, 3, 3.5]
weights = [0.2, 0.8, -0.5, 1.0, 4.0]
# there is only one value for the bias because it is a single neuron.
bias = 2


# for i in inputs:
#     for j in weights:
#         output = i * j + bias
        # output += i * j + bias
        # output += i * weights[inputs.index(i)]
        # output += bias
# print(output)
    # output += i * weights[inputs.index(i)]
    # output += bias
    # print(output)
    

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + inputs[4] * weights[4] + bias
print(output)