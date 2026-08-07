import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter3.dense_layer import Layer_Dense
from nnfs.datasets import spiral_data


inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]


# Reusable ReLU activation class, applied after a Dense layer's forward
# pass to introduce non-linearity (without it, stacked Dense layers would
# collapse into an equivalent single linear layer).
class Activation_ReLu:

    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
    
activation1 = Activation_ReLu()

dense1.forward(X)

activation1.forward(dense1.output)

print(activation1.output[:5])