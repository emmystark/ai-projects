# Chapter 4: ReLU (Rectified Linear Unit) activation — passes positive
# values through unchanged and clips negative values to 0. Shown first as
# a manual loop, then with numpy's np.maximum for the same result.
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

outputs = []
# for i in inputs:
#     if i > 0:
#         outputs.append(i)
#     else:
#         outputs.append(0)
        
for i in inputs:
    outputs.append(max(i, 0))
print(outputs)


# using numpy for the same tasks

import numpy as np

outputs = np.maximum(0, inputs)
print(outputs)