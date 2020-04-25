import numpy as np

weights = [[1.1, 2.3, 5.1, -2.0],
           [2.2, 1.7, -4.1, 3.7],
           [3.3, -4.2, 1.3, 2.0]]
biases = [2, 1.2, 0.5]  
inputs = [1, 2, 3, 4]


output = np.dot(weights, inputs) + biases
print(output)
