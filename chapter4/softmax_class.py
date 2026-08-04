import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter3.dense_layer import Layer_Dense

from chapter4.ReLu_class import Activation_ReLu


from chapter5.cross_entropy_loss import Loss_CategoricalCrossentropy

from nnfs.datasets import spiral_data

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]


class Activation_Softmax:
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    
    

print(np.exp(1))
print(np.exp(10))
# print(np.exp(1000))


# Preventing overflow in softmax function e.g in print(np.exp(1000))


print(np.exp(-np.inf), np.exp(0))


softmax = Activation_Softmax()

softmax.forward([[1, 2, 3]])

# substracted_softmax = Activation_Softmax()


print(softmax.output)

# subtracted 3 - max from the list
substracted_softmax = softmax.forward([[-2, -1, 0]])

print(substracted_softmax)

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])


loss = loss_function.calculate(activation2.output, y)

print(f"Loss: {loss}")


predictions = np.argmax(activation2.output, axis=1)

if len(y.shape) == 2:
    y. np.argmax(y, axis=1)
    
accuracy = np.mean(predictions==y)


print('acc', accuracy)