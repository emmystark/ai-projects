# Manual dot product of two 3-element vectors, done element-by-element
# before switching to numpy's np.dot in later scripts.
a = [1, 2, 3]
b = [2, 3, 4]

dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

print(dot_product)