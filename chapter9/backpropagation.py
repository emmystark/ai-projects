# Chapter 9: manual backpropagation through a single neuron (3 inputs -> ReLU),
# computing every intermediate gradient by hand via the chain rule, then using
# those gradients to nudge weights/bias and confirm the output (post-ReLU) drops.

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0


# Forward Pass
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

print(xw0, xw1, xw2)

z = xw0 + xw1 + xw2 + b

print('-----')
print(z)

# ReLU activation function
y = max(z, 0)
print(y)

# Gradient flowing in from "downstream" (e.g. the next layer/loss); assumed
# to be 1.0 here since this neuron is treated as the final output.
dvalue = 1.0

relu_dz = (1. if z > 0 else 0.)
# Derivative of ReLU w.r.t. z, chained with the incoming gradient (dvalue).
drelu_dz = dvalue * (1. if z > 0 else 0.)


# Derivative of a sum w.r.t. each of its addends is always 1.
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

# Derivative of x*w w.r.t. x is w, and w.r.t. w is x.
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]


# Chain the ReLU gradient back through the sum (z = xw0+xw1+xw2+b).
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db


# Chain further back through each multiplication to get gradients
# w.r.t. inputs (x) and weights (w).
drelu_x0 = drelu_dxw0 * dmul_dx0
drelu_x1 = drelu_dxw1 * dmul_dx1
drelu_x2 = drelu_dxw2 * dmul_dx2

drelu_w0 = drelu_dxw0 * dmul_dw0
drelu_w1 = drelu_dxw1 * dmul_dw1
drelu_w2 = drelu_dxw2 * dmul_dw2


dx = [drelu_x0, drelu_x1, drelu_x2]
dw = [drelu_w0, drelu_w1, drelu_w2]
db = drelu_db


# Gradient descent step: nudge weights/bias slightly in the negative
# gradient direction (small learning rate of 0.001) to reduce the output.
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

# Re-run the forward pass with updated weights/bias to confirm the
# ReLU output decreased, as expected from a gradient descent step.
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]


z = xw0 + xw1 + xw2 + b

y = max(z, 0)

print(y)

print(w, b)

print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

print('---------')

# print(drelu_x0, drelu_x1, drelu_x2, drelu_w0, drelu_w1, drelu_w2)


print(drelu_dz)
# print(xw0)

