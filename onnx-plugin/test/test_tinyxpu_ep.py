"""
Test script demonstrating how to load and use the Sample EP plugin.
Compatible with ONNX Runtime 1.23+

Author: Avik De
Licensed under the MIT License.

Usage:
    python test_sample_ep.py [path_to_libsample_ep.so]
"""
import sys
import os

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def build_test_model():
    """Build an ONNX model with Add, Sub, Mul, Div ops."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    Z_add = helper.make_tensor_value_info("Z_add", TensorProto.FLOAT, [1, 4])
    Z_sub = helper.make_tensor_value_info("Z_sub", TensorProto.FLOAT, [1, 4])
    Z_mul = helper.make_tensor_value_info("Z_mul", TensorProto.FLOAT, [1, 4])
    Z_div = helper.make_tensor_value_info("Z_div", TensorProto.FLOAT, [1, 4])

    nodes = [
        helper.make_node("Add", ["X", "Y"], ["Z_add"], name="add_node"),
        helper.make_node("Sub", ["X", "Y"], ["Z_sub"], name="sub_node"),
        helper.make_node("Mul", ["X", "Y"], ["Z_mul"], name="mul_node"),
        helper.make_node("Div", ["X", "Y"], ["Z_div"], name="div_node"),
    ]

    graph = helper.make_graph(nodes, "test_graph", [X, Y],
                              [Z_add, Z_sub, Z_mul, Z_div])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def main():
    print(f"ONNX Runtime Version: {ort.__version__}")
    print(f"ONNX Runtime loaded successfully\n")

    # Determine plugin library path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "..", "build", "libsample_ep.so")
    plugin_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    plugin_path = os.path.abspath(plugin_path)

    print(f"Registering plugin EP from: {plugin_path}")
    ort.register_execution_provider_library("SampleEP", plugin_path)
    print("Plugin EP registered successfully!\n")

    # =========================================================================
    # Query and display EP device information
    # =========================================================================
    ep_devices = ort.get_ep_devices()
    sample_ep_devices = []

    print(f"Found {len(ep_devices)} EP device(s):\n")
    for i, ep_device in enumerate(ep_devices):
        hw = ep_device.device
        hw_type = ort.OrtHardwareDeviceType(hw.type)

        print(f"  EP Device {i}:")
        print(f"    Name:           {ep_device.ep_name}")
        print(f"    Vendor:         {ep_device.ep_vendor}")
        print(f"    HW Device Type: {hw_type.name}")
        print(f"    HW Vendor:      {hw.vendor}")
        print(f"    HW Vendor ID:   0x{hw.vendor_id:04x}")
        print(f"    HW Device ID:   0x{hw.device_id:04x}")
        print()

        if "SampleEP" in ep_device.ep_name:
            sample_ep_devices.append(ep_device)

    if not sample_ep_devices:
        print("ERROR: Could not find SampleEP device")
        return 1

    # =========================================================================
    # Build a test model and create a session to discover supported ops
    # =========================================================================
    print("Building test model with ops: Add, Sub, Mul, Div...")
    model_bytes = build_test_model()

    # Create session with the plugin EP to trigger GetCapability
    session_options = ort.SessionOptions()
    session_options.add_provider_for_devices(sample_ep_devices, {})

    print("\nCreating session (EP will report claimed ops):")
    sys.stdout.flush()
    session = ort.InferenceSession(model_bytes, sess_options=session_options)
    sys.stdout.flush()
    print("Session created successfully!\n")

    # Run inference to verify the EP works
    print("Running inference...")
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

    results = session.run(None, {"X": x, "Y": y})
    z_add, z_sub, z_mul, z_div = results

    print(f"  X       = {x[0]}")
    print(f"  Y       = {y[0]}")
    print(f"  X + Y   = {z_add[0]}  (Add - handled by SampleEP)")
    print(f"  X - Y   = {z_sub[0]}  (Sub - handled by CPU)")
    print(f"  X * Y   = {z_mul[0]}  (Mul - handled by SampleEP)")
    print(f"  X / Y   = {z_div[0]}  (Div - handled by CPU)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    del session
    print("\nUnregistering plugin EP...")
    ort.unregister_execution_provider_library("SampleEP")
    print("Plugin EP unregistered successfully")

    print("\nTest completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
