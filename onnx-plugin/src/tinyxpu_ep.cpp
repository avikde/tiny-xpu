// Copyright (c) Sample EP Authors. Licensed under the MIT License.
// Sample ONNX Runtime Execution Provider Plugin Implementation
// Compatible with ONNX Runtime 1.22+

#include "tinyxpu_ep.h"
#include <cstring>
#include <cstddef>
#include <cstdio>

// Platform-specific export macro
#if defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#elif defined(__APPLE__) || defined(__linux__)
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

// Helper macro to get containing object from member pointer (like container_of in Linux kernel)
#define CONTAINER_OF(ptr, type, member) \
    reinterpret_cast<type*>(reinterpret_cast<char*>(ptr) - offsetof(type, member))

#define CONTAINER_OF_CONST(ptr, type, member) \
    reinterpret_cast<const type*>(reinterpret_cast<const char*>(ptr) - offsetof(type, member))

// Global API pointers (initialized in CreateEpFactories)
static ApiPtrs g_apis;

// ============================================================================
// Exported Plugin Entry Points
// ============================================================================

extern "C" {

EXPORT_SYMBOL OrtStatus* ORT_API_CALL CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories) noexcept {

    // Initialize global API pointers
    g_apis.Init(ort_api_base, default_logger);

    if (max_factories < 1) {
        return g_apis.ort_api->CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Need space for at least 1 factory");
    }

    // Create our factory
    auto* factory = new SampleEpFactory(registration_name, g_apis);
    factories[0] = factory->GetOrtFactory();
    *num_factories = 1;

    return nullptr;  // Success
}

EXPORT_SYMBOL OrtStatus* ORT_API_CALL ReleaseEpFactory(OrtEpFactory* factory) noexcept {
    auto* sample_factory = SampleEpFactory::FromOrt(factory);
    delete sample_factory;
    return nullptr;  // Success
}

}  // extern "C"

// ============================================================================
// SampleEpFactory Implementation
// ============================================================================

SampleEpFactory* SampleEpFactory::FromOrt(OrtEpFactory* ort_factory) {
    return CONTAINER_OF(ort_factory, SampleEpFactory, factory_);
}

const SampleEpFactory* SampleEpFactory::FromOrt(const OrtEpFactory* ort_factory) {
    return CONTAINER_OF_CONST(ort_factory, SampleEpFactory, factory_);
}

SampleEpFactory::SampleEpFactory(const char* name, const ApiPtrs& apis)
    : ep_name_(std::string(name) + "PluginExecutionProvider"), apis_(apis) {

    // Zero-initialize the OrtEpFactory struct
    std::memset(&factory_, 0, sizeof(factory_));

    // Initialize the OrtEpFactory vtable
    factory_.ort_version_supported = ORT_API_VERSION;

    // Required callbacks
    factory_.GetName = GetNameImpl;
    factory_.GetVendor = GetVendorImpl;
    factory_.GetSupportedDevices = GetSupportedDevicesImpl;
    factory_.CreateEp = CreateEpImpl;
    factory_.ReleaseEp = ReleaseEpImpl;

    // Version 1.23 additions
    factory_.GetVendorId = GetVendorIdImpl;
    factory_.GetVersion = GetVersionImpl;
    factory_.ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
    factory_.CreateAllocator = CreateAllocatorImpl;
    factory_.ReleaseAllocator = ReleaseAllocatorImpl;
    factory_.CreateDataTransfer = CreateDataTransferImpl;
    factory_.IsStreamAware = IsStreamAwareImpl;
    factory_.CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

SampleEpFactory::~SampleEpFactory() = default;

const char* ORT_API_CALL SampleEpFactory::GetNameImpl(const OrtEpFactory* this_) noexcept {
    return FromOrt(this_)->ep_name_.c_str();
}

const char* ORT_API_CALL SampleEpFactory::GetVendorImpl(const OrtEpFactory* this_) noexcept {
    (void)this_;
    return "SampleVendor";
}

uint32_t ORT_API_CALL SampleEpFactory::GetVendorIdImpl(const OrtEpFactory* this_) noexcept {
    (void)this_;
    return 0x1234;  // Sample vendor ID
}

const char* ORT_API_CALL SampleEpFactory::GetVersionImpl(const OrtEpFactory* this_) noexcept {
    (void)this_;
    return "1.0.0";
}

OrtStatus* ORT_API_CALL SampleEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* num_ep_devices) noexcept {

    auto* factory = FromOrt(this_);
    const auto& apis = factory->GetApis();

    *num_ep_devices = 0;

    // Look through available hardware devices and claim CPU devices
    for (size_t i = 0; i < num_devices && *num_ep_devices < max_ep_devices; ++i) {
        const OrtHardwareDevice* hw_device = devices[i];

        // Use the API to get the device type (OrtHardwareDevice is opaque)
        OrtHardwareDeviceType device_type = apis.ort_api->HardwareDevice_Type(hw_device);

        // For this sample, we support CPU devices
        if (device_type == OrtHardwareDeviceType_CPU) {
            OrtEpDevice* ep_device = nullptr;
            OrtStatus* status = apis.ep_api->CreateEpDevice(
                this_,
                hw_device,
                nullptr,  // ep_metadata
                nullptr,  // ep_options
                &ep_device);

            if (status != nullptr) {
                return status;
            }

            ep_devices[*num_ep_devices] = ep_device;
            (*num_ep_devices)++;
        }
    }

    return nullptr;  // Success
}

OrtStatus* ORT_API_CALL SampleEpFactory::CreateEpImpl(
    OrtEpFactory* this_,
    const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* ep_metadata_pairs,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {

    (void)devices;
    (void)ep_metadata_pairs;
    (void)num_devices;
    (void)session_options;

    auto* factory = FromOrt(this_);
    auto* sample_ep = new SampleEp(factory, logger);
    *ep = sample_ep->GetOrtEp();
    return nullptr;
}

void ORT_API_CALL SampleEpFactory::ReleaseEpImpl(OrtEpFactory* this_, OrtEp* ep) noexcept {
    (void)this_;
    auto* sample_ep = SampleEp::FromOrt(ep);
    delete sample_ep;
}

OrtStatus* ORT_API_CALL SampleEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    const char* compatibility_info,
    OrtCompiledModelCompatibility* model_compatibility) noexcept {
    (void)this_;
    (void)devices;
    (void)num_devices;
    (void)compatibility_info;
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* allocator_options,
    OrtAllocator** allocator) noexcept {
    (void)this_;
    (void)memory_info;
    (void)allocator_options;
    *allocator = nullptr;  // Use default CPU allocator
    return nullptr;
}

void ORT_API_CALL SampleEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* this_, OrtAllocator* allocator) noexcept {
    (void)this_;
    (void)allocator;
}

OrtStatus* ORT_API_CALL SampleEpFactory::CreateDataTransferImpl(
    OrtEpFactory* this_,
    OrtDataTransferImpl** data_transfer) noexcept {
    (void)this_;
    *data_transfer = nullptr;  // No custom data transfer needed for CPU EP
    return nullptr;
}

bool ORT_API_CALL SampleEpFactory::IsStreamAwareImpl(const OrtEpFactory* this_) noexcept {
    (void)this_;
    return false;
}

OrtStatus* ORT_API_CALL SampleEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_,
    const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* stream_options,
    OrtSyncStreamImpl** stream) noexcept {
    (void)this_;
    (void)memory_device;
    (void)stream_options;
    *stream = nullptr;
    return nullptr;
}

// ============================================================================
// SampleEp Implementation
// ============================================================================

SampleEp* SampleEp::FromOrt(OrtEp* ort_ep) {
    return CONTAINER_OF(ort_ep, SampleEp, ep_);
}

const SampleEp* SampleEp::FromOrt(const OrtEp* ort_ep) {
    return CONTAINER_OF_CONST(ort_ep, SampleEp, ep_);
}

SampleEp::SampleEp(SampleEpFactory* factory, const OrtLogger* session_logger)
    : factory_(factory), session_logger_(session_logger) {

    // Zero-initialize the OrtEp struct
    std::memset(&ep_, 0, sizeof(ep_));

    // Initialize the OrtEp vtable
    ep_.ort_version_supported = ORT_API_VERSION;

    // Required callbacks
    ep_.GetName = GetNameImpl;
    ep_.GetCapability = GetCapabilityImpl;
    ep_.Compile = CompileImpl;
    ep_.ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;

    // 1.23 optional callback stubs
    ep_.GetPreferredDataLayout = GetPreferredDataLayoutImpl;
    ep_.ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
    ep_.SetDynamicOptions = SetDynamicOptionsImpl;
    ep_.OnRunStart = OnRunStartImpl;
    ep_.OnRunEnd = OnRunEndImpl;
    ep_.CreateAllocator = EpCreateAllocatorImpl;
    ep_.CreateSyncStreamForDevice = EpCreateSyncStreamForDeviceImpl;
    ep_.GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;
}

SampleEp::~SampleEp() = default;

const char* ORT_API_CALL SampleEp::GetNameImpl(const OrtEp* this_) noexcept {
    return FromOrt(this_)->factory_->GetEpName().c_str();
}

OrtStatus* ORT_API_CALL SampleEp::GetCapabilityImpl(
    OrtEp* this_,
    const OrtGraph* graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {

    auto* ep = FromOrt(this_);
    const auto& apis = ep->GetApis();

    // Get all nodes in the graph
    size_t num_nodes = 0;
    OrtStatus* status = apis.ort_api->Graph_GetNumNodes(graph, &num_nodes);
    if (status != nullptr) {
        return status;
    }

    if (num_nodes == 0) {
        return nullptr;  // No nodes to process
    }

    // Get all nodes from the graph
    std::vector<const OrtNode*> all_nodes(num_nodes);
    status = apis.ort_api->Graph_GetNodes(graph, all_nodes.data(), num_nodes);
    if (status != nullptr) {
        return status;
    }

    // Check each node and claim supported operators
    for (size_t i = 0; i < num_nodes; ++i) {
        const OrtNode* node = all_nodes[i];
        if (!node) continue;

        const char* op_type = nullptr;
        status = apis.ort_api->Node_GetOperatorType(node, &op_type);
        if (status != nullptr) {
            return status;
        }

        // Support Add and Mul operators as an example
        if (op_type && (std::strcmp(op_type, "Add") == 0 ||
                       std::strcmp(op_type, "Mul") == 0)) {
            printf("  [SampleEP] Claiming op: %s\n", op_type);
            fflush(stdout);

            // Add each supported node as its own fused group
            const OrtNode* single_node = node;
            status = apis.ep_api->EpGraphSupportInfo_AddNodesToFuse(
                graph_support_info,
                &single_node,
                1,
                nullptr);

            if (status != nullptr) {
                return status;
            }
        }
    }

    return nullptr;  // Success
}

OrtStatus* ORT_API_CALL SampleEp::CompileImpl(
    OrtEp* this_,
    const OrtGraph** graphs,
    const OrtNode** fused_nodes,
    size_t count,
    OrtNodeComputeInfo** node_compute_infos,
    OrtNode** ep_context_nodes) noexcept {

    (void)graphs;
    (void)fused_nodes;

    auto* ep = FromOrt(this_);
    const auto& apis = ep->GetApis();

    // Create a compute info for each fused graph
    for (size_t i = 0; i < count; ++i) {
        auto* compute_info = new SampleNodeComputeInfo(apis);
        node_compute_infos[i] = compute_info->GetOrtComputeInfo();

        // Set ep_context_nodes to nullptr since we don't support EPContext models
        if (ep_context_nodes) {
            ep_context_nodes[i] = nullptr;
        }
    }

    return nullptr;  // Success
}

void ORT_API_CALL SampleEp::ReleaseNodeComputeInfosImpl(
    OrtEp* this_,
    OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {

    (void)this_;

    for (size_t i = 0; i < num_node_compute_infos; ++i) {
        auto* sample_info = SampleNodeComputeInfo::FromOrt(node_compute_infos[i]);
        delete sample_info;
    }
}

OrtStatus* ORT_API_CALL SampleEp::GetPreferredDataLayoutImpl(
    OrtEp* this_, OrtEpDataLayout* preferred_data_layout) noexcept {
    (void)this_;
    *preferred_data_layout = OrtEpDataLayout::OrtEpDataLayout_NCHW;
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::ShouldConvertDataLayoutForOpImpl(
    OrtEp* this_, const char* domain, const char* op_type,
    OrtEpDataLayout target_data_layout, int* should_convert) noexcept {
    (void)this_;
    (void)domain;
    (void)op_type;
    (void)target_data_layout;
    *should_convert = -1;  // Let ORT decide
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::SetDynamicOptionsImpl(
    OrtEp* this_, const char* const* option_keys,
    const char* const* option_values, size_t num_options) noexcept {
    (void)this_;
    (void)option_keys;
    (void)option_values;
    (void)num_options;
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::OnRunStartImpl(
    OrtEp* this_, const OrtRunOptions* run_options) noexcept {
    (void)this_;
    (void)run_options;
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::OnRunEndImpl(
    OrtEp* this_, const OrtRunOptions* run_options, bool sync_stream) noexcept {
    (void)this_;
    (void)run_options;
    (void)sync_stream;
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::EpCreateAllocatorImpl(
    OrtEp* this_, const OrtMemoryInfo* memory_info,
    OrtAllocator** allocator) noexcept {
    (void)this_;
    (void)memory_info;
    *allocator = nullptr;  // Use default
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::EpCreateSyncStreamForDeviceImpl(
    OrtEp* this_, const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) noexcept {
    (void)this_;
    (void)memory_device;
    *stream = nullptr;
    return nullptr;
}

const char* ORT_API_CALL SampleEp::GetCompiledModelCompatibilityInfoImpl(
    OrtEp* this_, const OrtGraph* graph) noexcept {
    (void)this_;
    (void)graph;
    return nullptr;
}

// ============================================================================
// SampleNodeComputeInfo Implementation
// ============================================================================

SampleNodeComputeInfo* SampleNodeComputeInfo::FromOrt(OrtNodeComputeInfo* ort_info) {
    return CONTAINER_OF(ort_info, SampleNodeComputeInfo, compute_info_);
}

SampleNodeComputeInfo::SampleNodeComputeInfo(const ApiPtrs& apis)
    : ort_api(apis.ort_api), ep_api(apis.ep_api) {

    // Zero-initialize the OrtNodeComputeInfo struct
    std::memset(&compute_info_, 0, sizeof(compute_info_));

    compute_info_.ort_version_supported = ORT_API_VERSION;

    compute_info_.CreateState = CreateStateImpl;
    compute_info_.Compute = ComputeImpl;
    compute_info_.ReleaseState = ReleaseStateImpl;
}

// Simple compute state - just stores a flag
struct ComputeState {
    bool initialized = true;
};

OrtStatus* ORT_API_CALL SampleNodeComputeInfo::CreateStateImpl(
    OrtNodeComputeInfo* this_,
    OrtNodeComputeContext* compute_context,
    void** compute_state) noexcept {

    (void)this_;
    (void)compute_context;

    // Create compute state
    *compute_state = new ComputeState();
    return nullptr;
}

OrtStatus* ORT_API_CALL SampleNodeComputeInfo::ComputeImpl(
    OrtNodeComputeInfo* this_,
    void* compute_state,
    OrtKernelContext* kernel_context) noexcept {

    auto* info = FromOrt(this_);
    (void)compute_state;

    // Get input tensors
    const OrtValue* input_0 = nullptr;
    const OrtValue* input_1 = nullptr;
    OrtStatus* status = info->ort_api->KernelContext_GetInput(kernel_context, 0, &input_0);
    if (status != nullptr) return status;
    status = info->ort_api->KernelContext_GetInput(kernel_context, 1, &input_1);
    if (status != nullptr) return status;

    if (!input_0 || !input_1) {
        return info->ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "Missing inputs");
    }

    // Get tensor info
    OrtTensorTypeAndShapeInfo* input_info = nullptr;
    status = info->ort_api->GetTensorTypeAndShape(input_0, &input_info);
    if (status != nullptr) return status;

    size_t num_dims = 0;
    status = info->ort_api->GetDimensionsCount(input_info, &num_dims);
    if (status != nullptr) {
        info->ort_api->ReleaseTensorTypeAndShapeInfo(input_info);
        return status;
    }

    std::vector<int64_t> dims(num_dims);
    status = info->ort_api->GetDimensions(input_info, dims.data(), num_dims);
    if (status != nullptr) {
        info->ort_api->ReleaseTensorTypeAndShapeInfo(input_info);
        return status;
    }

    size_t total_elements = 1;
    for (auto d : dims) total_elements *= static_cast<size_t>(d);

    info->ort_api->ReleaseTensorTypeAndShapeInfo(input_info);

    // Create output tensor
    OrtValue* output = nullptr;
    status = info->ort_api->KernelContext_GetOutput(kernel_context, 0, dims.data(), num_dims, &output);
    if (status != nullptr) return status;

    if (!output) {
        return info->ort_api->CreateStatus(ORT_FAIL, "Failed to create output");
    }

    // Get data pointers (assuming float tensors for this sample)
    const float* data_0 = nullptr;
    const float* data_1 = nullptr;
    float* data_out = nullptr;

    status = info->ort_api->GetTensorData(input_0, (const void**)&data_0);
    if (status != nullptr) return status;
    status = info->ort_api->GetTensorData(input_1, (const void**)&data_1);
    if (status != nullptr) return status;
    status = info->ort_api->GetTensorMutableData(output, (void**)&data_out);
    if (status != nullptr) return status;

    // Perform element-wise addition (even for Mul, just for demonstration)
    // In a real EP, this would dispatch to hardware
    for (size_t i = 0; i < total_elements; ++i) {
        data_out[i] = data_0[i] + data_1[i];
    }

    return nullptr;  // Success
}

void ORT_API_CALL SampleNodeComputeInfo::ReleaseStateImpl(
    OrtNodeComputeInfo* this_,
    void* compute_state) noexcept {

    (void)this_;
    delete static_cast<ComputeState*>(compute_state);
}
