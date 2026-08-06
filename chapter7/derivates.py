# Chapter 7: estimating derivatives numerically using the slope between
# two very close points on f(x) = 2x^2 (the "delta method"), then drawing
# the resulting tangent lines to visually confirm they match the curve.
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # return 2*x
    return 2*x**2

x = np.array(np.arange(0,5,0.001))
y = f(x)

# print(x)
# print(y)


plt.plot(x,y)

colors = ['k', 'g', 'r', 'b', 'c']


# plt.show()

# Crude slope estimate using just the first two sampled points.
print([(y[1] - y[0]) / (x[1] - x[0])])

# Tiny step size used to approximate the derivative at a single point (x1=2)
# via (f(x1+delta) - f(x1)) / delta.
p2_delta = 0.0001

x1 = 2
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

approximate_derivative = (y2-y1)/(x2-x1)

# y-intercept (b) of the tangent line at x1, from y = mx + b.
b = y2 - approximate_derivative*x2

# print(approximate_derivative)


def tangent_line(x):
    return approximate_derivative*x + b


to_plot = [x1-0.9, x1, x1+0.9]

plt.plot(to_plot, [tangent_line(i) for i in to_plot])


print('Approximate derivate for f(x)', f'where x = {x1} is {approximate_derivative}')


# plt.show()


def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative*x) + b


# Repeat the same numerical-derivative + tangent-line process for x = 0..4,
# plotting each point and its tangent line in a different color to show
# how the slope changes as x increases (since f is quadratic).
for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1+p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1, y1), (x2, y2))
    approximate_derivative = (y2-y1)/(x2-x1)
    b = y2-(approximate_derivative*x2)

    to_plot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot([point for point in to_plot], [approximate_tangent_line(point, approximate_derivative)
                                 for point in to_plot],
             c=colors[i])

    print('Approximate derivative for f(x)',
          f'where x = {x1} is {approximate_derivative}')


    plt.show()