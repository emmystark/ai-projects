# Chapter 9: backward pass (gradient) for the Softmax activation.
#
# Unlike ReLU, each Softmax output S_i depends on every input z_j in its
# sample (not just z_i), because of the normalizing sum in the denominator.
# So the derivative of one sample's output w.r.t. its inputs is a full
# Jacobian matrix, not a single per-element value:
#
#   dS_i/dz_j = S_i * (delta_ij - S_j)
#
# where delta_ij is 1 when i == j and 0 otherwise (Kronecker delta).
# In matrix form for one sample: J = diagflat(S) - S @ S.T

import numpy as np


class Activation_Softmax:

    def forward(self, inputs):
        self.inputs = inputs
        # Subtract the max per-sample before exponentiating to avoid
        # overflow (exp of large numbers); this doesn't change the result
        # since it cancels out in the normalization below.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # One gradient row per sample, same shape as the output.
        self.dinputs = np.empty_like(dvalues)

        # Softmax's Jacobian is per-sample, so it can't be vectorized across
        # the whole batch in one matrix op — loop over samples.
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten to a column vector so the outer product below works.
            single_output = single_output.reshape(-1, 1)

            # diagflat(S) handles the i == j terms (S_i * (1 - S_i)),
            # and S @ S.T handles the i != j terms (-S_i * S_j).
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Chain rule: multiply the Jacobian by the incoming gradient
            # (dvalues) from the next layer/loss to get this sample's
            # gradient w.r.t. its inputs.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


if __name__ == "__main__":
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                 [0.1, 0.5, 0.4],
                                 [0.02, 0.9, 0.08]])

    class_targets = np.array([0, 1, 1])

    softmax = Activation_Softmax()
    softmax.output = softmax_outputs

    # Fake an incoming gradient (as if from the loss) for demonstration.
    dvalues = softmax_outputs.copy()
    dvalues[range(len(dvalues)), class_targets] -= 1

    softmax.backward(dvalues)
    print(softmax.dinputs)
