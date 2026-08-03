import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]


b = np.array([b]).T

product_of_a_and_b_transpose = np.dot(a, b)

row_matrix = np.array([a])

# alternate method to create a row matrix

row_matrix2 = np.expand_dims(np.array(a), axis=0)

print(row_matrix)

print(row_matrix2)

print(product_of_a_and_b_transpose)
