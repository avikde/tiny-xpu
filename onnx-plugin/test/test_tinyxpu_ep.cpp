/**
 * Test application demonstrating how to load and use the Sample EP plugin
 * Compatible with ONNX Runtime 1.23+
 * Author: Avik De
 * Licensed under the MIT License.
 */
#include <onnxruntime_c_api.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper macro to check ORT status
#define CHECK_STATUS(expr)                                              \
    do {                                                                \
        OrtStatus* status = (expr);                                     \
        if (status != nullptr) {                                        \
            const char* msg = g_ort->GetErrorMessage(status);           \
            std::cerr << "Error: " << msg << std::endl;                 \
            g_ort->ReleaseStatus(status);                               \
            return 1;                                                   \
        }                                                               \
    } while (0)

const OrtApi* g_ort = nullptr;

int main(int argc, char* argv[]) {
    // Get the ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        std::cerr << "Failed to get ORT API" << std::endl;
        return 1;
    }

    std::cout << "ONNX Runtime Version: " << OrtGetApiBase()->GetVersionString() << std::endl;
    std::cout << "ORT API Version: " << ORT_API_VERSION << std::endl;

    // Create environment
    OrtEnv* env = nullptr;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test_sample_ep", &env));
    std::cout << "ONNX Runtime loaded successfully\n" << std::endl;

    // Determine the plugin library path
    const char* plugin_path = "./libsample_ep.so";
    if (argc > 1) {
        plugin_path = argv[1];
    }

    std::cout << "Registering plugin EP from: " << plugin_path << std::endl;

    // Register our plugin EP library
    OrtStatus* status = g_ort->RegisterExecutionProviderLibrary(env, "SampleEP", plugin_path);
    if (status != nullptr) {
        std::cerr << "RegisterExecutionProviderLibrary failed: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    std::cout << "Plugin EP registered successfully!\n" << std::endl;

    // =========================================================================
    // Query and display EP device information
    // =========================================================================
    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_ep_devices = 0;
    status = g_ort->GetEpDevices(env, &ep_devices, &num_ep_devices);
    if (status != nullptr) {
        std::cerr << "GetEpDevices failed: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return 1;
    }

    // Find our plugin EP device and print info for all devices
    const OrtEpDevice* sample_ep_device = nullptr;

    std::cout << "Found " << num_ep_devices << " EP device(s):\n" << std::endl;
    for (size_t i = 0; i < num_ep_devices; ++i) {
        const OrtEpDevice* ep_device = ep_devices[i];

        const char* ep_name = g_ort->EpDevice_EpName(ep_device);
        const char* ep_vendor = g_ort->EpDevice_EpVendor(ep_device);

        const OrtHardwareDevice* hw_device = g_ort->EpDevice_Device(ep_device);
        OrtHardwareDeviceType hw_type = g_ort->HardwareDevice_Type(hw_device);
        uint32_t hw_vendor_id = g_ort->HardwareDevice_VendorId(hw_device);
        uint32_t hw_device_id = g_ort->HardwareDevice_DeviceId(hw_device);
        const char* hw_vendor = g_ort->HardwareDevice_Vendor(hw_device);

        const char* type_str = "Unknown";
        switch (hw_type) {
            case OrtHardwareDeviceType_CPU: type_str = "CPU"; break;
            case OrtHardwareDeviceType_GPU: type_str = "GPU"; break;
            case OrtHardwareDeviceType_NPU: type_str = "NPU"; break;
            default: break;
        }

        std::cout << "  EP Device " << i << ":" << std::endl;
        std::cout << "    Name:           " << (ep_name ? ep_name : "(null)") << std::endl;
        std::cout << "    Vendor:         " << (ep_vendor ? ep_vendor : "(null)") << std::endl;
        std::cout << "    HW Device Type: " << type_str << std::endl;
        std::cout << "    HW Vendor:      " << (hw_vendor ? hw_vendor : "(null)") << std::endl;
        std::cout << "    HW Vendor ID:   0x" << std::hex << hw_vendor_id << std::dec << std::endl;
        std::cout << "    HW Device ID:   0x" << std::hex << hw_device_id << std::dec << std::endl;
        std::cout << std::endl;

        // Remember our plugin EP device for later
        if (ep_name && std::strstr(ep_name, "SampleEP") != nullptr) {
            sample_ep_device = ep_device;
        }
    }

    if (!sample_ep_device) {
        std::cerr << "Could not find SampleEP device" << std::endl;
        g_ort->ReleaseEnv(env);
        return 1;
    }

    // =========================================================================
    // Build a test model and create a session to discover supported ops
    // =========================================================================
    std::cout << "Building test model with ops: Add, Sub, Mul, Div..." << std::endl;

    const OrtModelEditorApi* model_api = g_ort->GetModelEditorApi();
    if (!model_api) {
        std::cerr << "ModelEditorApi not available (minimal build?)" << std::endl;
        g_ort->ReleaseEnv(env);
        return 1;
    }

    // Create tensor type info: float [1, 4]
    OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    CHECK_STATUS(g_ort->CreateTensorTypeAndShapeInfo(&tensor_info));
    CHECK_STATUS(g_ort->SetTensorElementType(tensor_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    int64_t shape[] = {1, 4};
    CHECK_STATUS(g_ort->SetDimensions(tensor_info, shape, 2));

    OrtTypeInfo* type_info = nullptr;
    CHECK_STATUS(model_api->CreateTensorTypeInfo(tensor_info, &type_info));
    g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);

    // Create graph inputs: X and Y
    OrtValueInfo* vi_x = nullptr;
    OrtValueInfo* vi_y = nullptr;
    CHECK_STATUS(model_api->CreateValueInfo("X", type_info, &vi_x));
    CHECK_STATUS(model_api->CreateValueInfo("Y", type_info, &vi_y));

    // Create graph outputs: one per op
    OrtValueInfo* vi_add = nullptr;
    OrtValueInfo* vi_sub = nullptr;
    OrtValueInfo* vi_mul = nullptr;
    OrtValueInfo* vi_div = nullptr;
    CHECK_STATUS(model_api->CreateValueInfo("Z_add", type_info, &vi_add));
    CHECK_STATUS(model_api->CreateValueInfo("Z_sub", type_info, &vi_sub));
    CHECK_STATUS(model_api->CreateValueInfo("Z_mul", type_info, &vi_mul));
    CHECK_STATUS(model_api->CreateValueInfo("Z_div", type_info, &vi_div));

    g_ort->ReleaseTypeInfo(type_info);

    // Create graph
    OrtGraph* graph = nullptr;
    CHECK_STATUS(model_api->CreateGraph(&graph));

    OrtValueInfo* inputs[] = {vi_x, vi_y};
    CHECK_STATUS(model_api->SetGraphInputs(graph, inputs, 2));

    OrtValueInfo* outputs[] = {vi_add, vi_sub, vi_mul, vi_div};
    CHECK_STATUS(model_api->SetGraphOutputs(graph, outputs, 4));

    // Create nodes for each op type
    const char* op_types[] = {"Add", "Sub", "Mul", "Div"};
    const char* out_names[] = {"Z_add", "Z_sub", "Z_mul", "Z_div"};
    const char* node_names[] = {"add_node", "sub_node", "mul_node", "div_node"};
    const char* in_names[] = {"X", "Y"};

    for (int i = 0; i < 4; ++i) {
        const char* out_name = out_names[i];
        OrtNode* node = nullptr;
        CHECK_STATUS(model_api->CreateNode(
            op_types[i], "", node_names[i],
            in_names, 2,
            &out_name, 1,
            nullptr, 0,
            &node));
        CHECK_STATUS(model_api->AddNodeToGraph(graph, node));
    }

    // Create model (ONNX domain, opset 13)
    const char* domains[] = {""};
    const int opsets[] = {13};
    OrtModel* model = nullptr;
    CHECK_STATUS(model_api->CreateModel(domains, opsets, 1, &model));
    CHECK_STATUS(model_api->AddGraphToModel(model, graph));

    // Create session options and attach our plugin EP
    OrtSessionOptions* session_options = nullptr;
    CHECK_STATUS(g_ort->CreateSessionOptions(&session_options));

    CHECK_STATUS(g_ort->SessionOptionsAppendExecutionProvider_V2(
        session_options, env,
        &sample_ep_device, 1,
        nullptr, nullptr, 0));

    // Create session - this triggers GetCapability and shows which ops the EP claims
    std::cout << "\nCreating session (EP will report claimed ops):" << std::endl;
    OrtSession* session = nullptr;
    status = model_api->CreateSessionFromModel(env, model, session_options, &session);

    g_ort->ReleaseModel(model);
    g_ort->ReleaseSessionOptions(session_options);

    if (status != nullptr) {
        std::cerr << "CreateSessionFromModel failed: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
    } else {
        std::cout << "Session created successfully!" << std::endl;
        g_ort->ReleaseSession(session);
    }

    // =========================================================================
    // Cleanup
    // =========================================================================
    std::cout << "\nUnregistering plugin EP..." << std::endl;
    status = g_ort->UnregisterExecutionProviderLibrary(env, "SampleEP");
    if (status != nullptr) {
        std::cerr << "UnregisterExecutionProviderLibrary failed: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    std::cout << "Plugin EP unregistered successfully" << std::endl;

    g_ort->ReleaseEnv(env);

    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
