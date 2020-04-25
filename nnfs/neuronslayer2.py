weights = [[1.1, 2.3, 5.1, -2.0],
           [2.2, 1.7, -4.1, 3.7],
           [3.3, -4.2, 1.3, 2.0]]
biases = [2, 1.2, 0.5]  
inputs = [1, 2, 3, 4]

layer_output = []
for node_weight, node_bias in zip(weights, biases):
    node_output = 0
    for weight, input in zip(node_weight, inputs):
        node_output = node_output + weight*input
    node_output = node_output + node_bias
    layer_output.append(node_output)     
    
print(layer_output) 
