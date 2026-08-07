# Illustrates increasing dimensionality: a 1D list (vector), a 2D list of
# lists (matrix), and a 3D list of lists of lists (tensor) — the shapes
# neural network data takes on as batches and layers stack up.
list = [1, 2, 3, 2.5]

listoflist = [[0.2, 0.8, -0.5, 1.0],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]]

listoflistoflist = [[[0.2, 0.8, -0.5, 1.0],
                     [0.5, -0.91, 0.26, -0.5],
                     [-0.26, -0.27, 0.17, 0.87]],
                    [[0.1, 0.2, 0.3, 0.4],
                     [0.5, 0.6, 0.7, 0.8],
                     [0.9, 1.0, 1.1, 1.2]]]

# listoflistoflist = listoflistoflist.shape()


print(listoflistoflist)