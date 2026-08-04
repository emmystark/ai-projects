import numpy as np
import nnfs
from nnfs.datasets import spiral
_
data
nnfs.init()
# Dense layer
class Layer
_
Dense :
# Layer initialization
def __
init
__ ( self , n
_
inputs , n
_
neurons ):
# Initialize weights and biases
self.weights = 0.01 * np.random.randn(n
_
inputs, n
_
self.biases = np.zeros(( 1 , n
_
neurons))
neurons)
# Forward pass
def forward ( self , inputs ):
# Calculate output values from inputs, weights and biases
self.output = np.dot(inputs, self.weights) + self.biases
# ReLU activation
class Activation
ReLU :
_
# Forward pass
def forward ( self , inputs ):
# Calculate output values from inputs
self.output = np.maximum( 0 , inputs)
# Softmax activation
class Activation
Softmax :
_
# Forward pass
def forward ( self , inputs ):
# Get unnormalized probabilities
exp_
values = np.exp(inputs - np.max(inputs, axis = 1 ,
keepdims = True ))
# Normalize them for each sample
probabilities = exp_
values / np.sum(exp_
values, axis = 1 ,
keepdims = True )
self.output = probabilities
# Create dataset
X, y = spiral
_
data( samples = 100 , classes = 3 )
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer
_
Dense( 2 , 3 )
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation
_
ReLU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer
_
Dense( 3 , 3 )
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation
_
Softmax()
# Make a forward pass of our training data through this layer
dense1.forward(X)
# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
# Let's see output of the first few samples:
print (activation2.output[: 5 ])