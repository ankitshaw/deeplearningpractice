weights = [[1.1, 2.3, 5.1, -2.0],
           [2.2, 1.7, -4.1, 3.7],
           [3.3, -4.2, 1.3, 2.0]]
biases = [2, 1.2, 0.5]  
inputs = [1, 2, 3, 4]

output = [weights[0][0]*inputs[0] + weights[0][1]*inputs[1] + weights[0][2]*inputs[2] + weights[0][3]*inputs[3] + biases[0],
          weights[1][0]*inputs[0] + weights[1][1]*inputs[1] + weights[1][2]*inputs[2] + weights[1][3]*inputs[3] + biases[1],
          weights[2][0]*inputs[0] + weights[2][1]*inputs[1] + weights[2][2]*inputs[2] + weights[2][3]*inputs[3] + biases[2]]

print(output)
    
