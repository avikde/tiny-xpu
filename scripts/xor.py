import onnx
from onnx import helper, TensorProto

# 1. Define Inputs and Outputs
# Input 'X' is a 1x2 vector
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
# Output 'Y' is a 1x1 scalar
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])

# 2. Define Weights and Biases (Pre-trained values for XOR)
# Hidden layer (2 inputs -> 2 outputs)
w1_vals = [1.0, 1.0, 1.0, 1.0] 
b1_vals = [0.0, -1.0]
# Output layer (2 inputs -> 1 output)
w2_vals = [1.0, -2.0]
b2_vals = [0.0]

# Create Initializer (Weight) Tensors
W1 = helper.make_tensor('W1', TensorProto.FLOAT, [2, 2], w1_vals)
B1 = helper.make_tensor('B1', TensorProto.FLOAT, [2], b1_vals)
W2 = helper.make_tensor('W2', TensorProto.FLOAT, [2, 1], w2_vals)
B2 = helper.make_tensor('B2', TensorProto.FLOAT, [1], b2_vals)

# 3. Create the Computation Nodes
# Node 1: MatMul (X * W1)
node1 = helper.make_node('MatMul', ['X', 'W1'], ['dot1'])
# Node 2: Add Bias
node2 = helper.make_node('Add', ['dot1', 'B1'], ['plus1'])
# Node 3: ReLU
node3 = helper.make_node('Relu', ['plus1'], ['relu1'])
# Node 4: MatMul (relu1 * W2)
node4 = helper.make_node('MatMul', ['relu1', 'W2'], ['dot2'])
# Node 5: Add Bias (Final Output)
node5 = helper.make_node('Add', ['dot2', 'B2'], ['Y'])

# 4. Build the Graph and Model
graph = helper.make_graph(
    [node1, node2, node3, node4, node5],
    'XOR_Network',
    [X], [Y],
    [W1, B1, W2, B2]
)

model = helper.make_model(graph, producer_name='onnx-example')

# 5. Save the model
onnx.save(model, 'xor_model.onnx')
print("Model saved as xor_model.onnx")

