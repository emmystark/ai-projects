# Demonstrates matrix/vector shape handling in numpy: turning a plain list
# into a column vector via .T, computing a matrix product, and two ways to
# turn a 1D list into a 2D "row matrix" (np.array([a]) vs np.expand_dims).
import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

# Wrap b in an extra list dimension, then transpose to get a column vector
# (shape (3, 1)) so np.dot(a, b) below is a valid matrix product.
b = np.array([b]).T

product_of_a_and_b_transpose = np.dot(a, b)

row_matrix = np.array([a])

# alternate method to create a row matrix

row_matrix2 = np.expand_dims(np.array(a), axis=0)

print(row_matrix)

print(row_matrix2)

print(product_of_a_and_b_transpose)
