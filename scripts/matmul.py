import onnx
from onnx import helper, TensorProto

# 1. Define Inputs and Outputs
# Input 'X' is a 2x3 matrix
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
# Output 'Y' is a 2x4 matrix
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4])

# 2. Define Weight
# W: 3x4
w_vals = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
]
W = helper.make_tensor('W', TensorProto.FLOAT, [3, 4], w_vals)

# 3. Create the Computation Node
node1 = helper.make_node('MatMul', ['X', 'W'], ['Y'])

# 4. Build the Graph and Model
graph = helper.make_graph(
    [node1],
    'MatMul_Network',
    [X], [Y],
    [W]
)

model = helper.make_model(graph, producer_name='onnx-example')

# 5. Save the model
onnx.save(model, 'matmul_model.onnx')
print("Model saved as matmul_model.onnx")
