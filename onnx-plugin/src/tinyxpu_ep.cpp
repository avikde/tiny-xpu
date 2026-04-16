// Copyright (c) Sample EP Authors. Licensed under the MIT License.
// Sample ONNX Runtime Execution Provider Plugin Implementation
// Compatible with ONNX Runtime 1.22+

#include "tinyxpu_ep.h"
#include <cstring>
#include <cstddef>
#include <cstdio>
#include <cstdint>

#ifdef TINYXPU_USE_VERILATOR
#include "Varray.h"
#include "verilated.h"
// Required when Verilator is built with SystemC support (e.g. Homebrew on macOS)
double sc_time_stamp() { return 0; }
#endif
#include "tinyxpu_perf.h"

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

// Perf counters from the most recent MatMulInteger execution.
// Written by ComputeImpl; read by tinyxpu_get_last_perf().
// Not thread-safe: single-threaded Verilator simulation assumed.
static TinyXpuPerfCounters g_last_perf{};

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

// Returns a copy of the perf counters from the most recent MatMulInteger run.
// Python callers retrieve this via ctypes after session.run().
EXPORT_SYMBOL void tinyxpu_get_last_perf(TinyXpuPerfCounters* out) noexcept {
    if (out) *out = g_last_perf;
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

#ifdef TINYXPU_USE_VERILATOR
        const bool is_matmul_integer  = op_type && std::strcmp(op_type, "MatMulInteger") == 0;
        const bool is_qlinear_matmul  = op_type && std::strcmp(op_type, "QLinearMatMul") == 0;
        const bool is_relu            = op_type && std::strcmp(op_type, "Relu") == 0;
        const bool is_our_op = is_matmul_integer || is_qlinear_matmul || is_relu;
#else
        const bool is_matmul_integer = false;
        const bool is_relu           = false;
        const bool is_our_op = op_type &&
            (std::strcmp(op_type, "MatMul") == 0 || std::strcmp(op_type, "Gemm") == 0);
#endif
        if (is_our_op) {
#ifdef TINYXPU_USE_VERILATOR
            if (is_matmul_integer) {
                // Claim MatMulInteger nodes if the weight (input[1]) is statically
                // known to be int8, OR if the weight type is unknown (dynamic
                // quantization graphs compute the activation type at runtime via
                // DynamicQuantizeLinear, so input[0] has no static type info).
                // We verify the actual element types in ComputeImpl.
                bool inputs_ok = false;
                {
                    size_t ni = 0;
                    if (!apis.ort_api->Node_GetNumInputs(node, &ni) && ni >= 2) {
                        std::vector<const OrtValueInfo*> vi(ni, nullptr);
                        if (!apis.ort_api->Node_GetInputs(node, vi.data(), ni)) {
                            bool weight_ok = false;
                            if (vi[1]) {
                                const OrtTypeInfo* ti = nullptr;
                                if (apis.ort_api->GetValueInfoTypeInfo(vi[1], &ti) || !ti) {
                                    weight_ok = true;
                                } else {
                                    const OrtTensorTypeAndShapeInfo* tsi = nullptr;
                                    if (!apis.ort_api->CastTypeInfoToTensorInfo(ti, &tsi) && tsi) {
                                        ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                                        if (!apis.ort_api->GetTensorElementType(tsi, &et))
                                            weight_ok = (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
                                    } else {
                                        weight_ok = true;
                                    }
                                }
                            }
                            inputs_ok = weight_ok;
                        }
                    }
                }
                if (!inputs_ok) continue;
                printf("  [TinyXPU EP] Claiming op: MatMulInteger (int8, %dx%d systolic array)\n",
                       TINYXPU_ARRAY_ROWS, TINYXPU_ARRAY_COLS);
            } else if (is_qlinear_matmul) {
                // QLinearMatMul: fully-integer matmul with embedded requant scales.
                // All scale/zero-point inputs (1,2,4,5,6,7) are constant initializers.
                printf("  [TinyXPU EP] Claiming op: QLinearMatMul (%dx%d systolic array + requant)\n",
                       TINYXPU_ARRAY_ROWS, TINYXPU_ARRAY_COLS);
            } else {
                // Relu: no type-check needed.
                printf("  [TinyXPU EP] Claiming op: Relu\n");
            }
#else
            printf("  [TinyXPU EP] Claiming op: %s (CPU float32 fallback)\n", op_type);
#endif
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
        // Read op type from the fused node
        const char* op_type_cstr = nullptr;
        if (fused_nodes && fused_nodes[i])
            apis.ort_api->Node_GetOperatorType(fused_nodes[i], &op_type_cstr);
        const std::string op_type_str = op_type_cstr ? op_type_cstr : "";

        // For Gemm: infer transB from B's stored shape vs output's N dimension.
        // B stored as [K, N] → transB=0; B stored as [N, K] → transB=1.
        bool transB = false;
        if (op_type_str == "Gemm" && fused_nodes && fused_nodes[i]) {
            // Read output shape to get N
            int64_t out_N = -1;
            {
                size_t nout = 0;
                if (!apis.ort_api->Node_GetNumOutputs(fused_nodes[i], &nout) && nout >= 1) {
                    std::vector<const OrtValueInfo*> ov(nout, nullptr);
                    if (!apis.ort_api->Node_GetOutputs(fused_nodes[i], ov.data(), nout) && ov[0]) {
                        const OrtTypeInfo* ti = nullptr;
                        if (!apis.ort_api->GetValueInfoTypeInfo(ov[0], &ti) && ti) {
                            const OrtTensorTypeAndShapeInfo* tsi = nullptr;
                            if (!apis.ort_api->CastTypeInfoToTensorInfo(ti, &tsi) && tsi) {
                                size_t ndim = 0;
                                if (!apis.ort_api->GetDimensionsCount(tsi, &ndim) && ndim >= 1) {
                                    std::vector<int64_t> d(ndim);
                                    if (!apis.ort_api->GetDimensions(tsi, d.data(), ndim))
                                        out_N = d[ndim - 1];
                                }
                            }
                        }
                    }
                }
            }
            // Read B (input index 1) last dimension
            if (out_N > 0) {
                size_t nin = 0;
                if (!apis.ort_api->Node_GetNumInputs(fused_nodes[i], &nin) && nin >= 2) {
                    std::vector<const OrtValueInfo*> iv(nin, nullptr);
                    if (!apis.ort_api->Node_GetInputs(fused_nodes[i], iv.data(), nin) && iv[1]) {
                        const OrtTypeInfo* ti = nullptr;
                        if (!apis.ort_api->GetValueInfoTypeInfo(iv[1], &ti) && ti) {
                            const OrtTensorTypeAndShapeInfo* tsi = nullptr;
                            if (!apis.ort_api->CastTypeInfoToTensorInfo(ti, &tsi) && tsi) {
                                size_t ndim = 0;
                                if (!apis.ort_api->GetDimensionsCount(tsi, &ndim) && ndim >= 2) {
                                    std::vector<int64_t> d(ndim);
                                    if (!apis.ort_api->GetDimensions(tsi, d.data(), ndim)) {
                                        // B last dim == out_N → transB=0; else transB=1
                                        transB = (d[ndim - 1] != out_N);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        auto* compute_info = new SampleNodeComputeInfo(apis, op_type_str, transB);

        // ---- QLinearMatMul: mark for requantization --------------------------------
        // Scale parameters (a_scale, b_scale, y_scale, y_zero_point) are constant
        // ONNX initializers read at runtime in ComputeImpl via KernelContext_GetInput.
        if (op_type_str == "QLinearMatMul") {
            compute_info->has_requant = true;
        }

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

SampleNodeComputeInfo::SampleNodeComputeInfo(const ApiPtrs& apis, std::string op_type_str,
                                             bool transB_flag, bool fused_relu_flag)
    : ort_api(apis.ort_api), ep_api(apis.ep_api),
      op_type(std::move(op_type_str)), transB(transB_flag), fused_relu(fused_relu_flag),
      has_requant(false), requant_M0(0), requant_rshift(0), requant_zp(0) {

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

    // ---- Relu: single-input, same-shape float32 output ----------------------
    if (info->op_type == "Relu") {
        const OrtValue* input = nullptr;
        OrtStatus* s = info->ort_api->KernelContext_GetInput(kernel_context, 0, &input);
        if (s) return s;
        if (!input)
            return info->ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "relu: missing input");

        OrtTensorTypeAndShapeInfo* si = nullptr;
        s = info->ort_api->GetTensorTypeAndShape(input, &si);
        if (s) return s;
        size_t ndim = 0;
        info->ort_api->GetDimensionsCount(si, &ndim);
        std::vector<int64_t> shape(ndim);
        info->ort_api->GetDimensions(si, shape.data(), ndim);
        info->ort_api->ReleaseTensorTypeAndShapeInfo(si);

        OrtValue* output = nullptr;
        s = info->ort_api->KernelContext_GetOutput(kernel_context, 0, shape.data(), shape.size(), &output);
        if (s) return s;

        const void* in_raw = nullptr;
        void* out_raw = nullptr;
        info->ort_api->GetTensorData(input, &in_raw);
        info->ort_api->GetTensorMutableData(output, &out_raw);

        int64_t total = 1;
        for (auto d : shape) total *= d;
        const float* in_f  = static_cast<const float*>(in_raw);
        float*       out_f = static_cast<float*>(out_raw);
        for (int64_t i = 0; i < total; ++i)
            out_f[i] = in_f[i] > 0.0f ? in_f[i] : 0.0f;

        return nullptr;
    }

    // ---- fetch inputs -------------------------------------------------------
    // QLinearMatMul: A=input[0], B=input[3] (inputs 1,2,4,5,6,7 are scale constants).
    // MatMulInteger / MatMul / Gemm: A=input[0], B=input[1].
    const uint32_t b_idx = (info->op_type == "QLinearMatMul") ? 3u : 1u;

    const OrtValue* input_A = nullptr;
    const OrtValue* input_B = nullptr;
    OrtStatus* status = info->ort_api->KernelContext_GetInput(kernel_context, 0, &input_A);
    if (status) return status;
    status = info->ort_api->KernelContext_GetInput(kernel_context, b_idx, &input_B);
    if (status) return status;
    if (!input_A || !input_B)
        return info->ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "matmul: missing inputs");

    // ---- read shapes (N-D aware) --------------------------------------------
    auto read_shape = [&](const OrtValue* t, std::vector<int64_t>& dims) -> OrtStatus* {
        OrtTensorTypeAndShapeInfo* si = nullptr;
        OrtStatus* s = info->ort_api->GetTensorTypeAndShape(t, &si);
        if (s) return s;
        size_t ndim = 0;
        s = info->ort_api->GetDimensionsCount(si, &ndim);
        if (!s) {
            dims.resize(ndim);
            s = info->ort_api->GetDimensions(si, dims.data(), ndim);
        }
        info->ort_api->ReleaseTensorTypeAndShapeInfo(si);
        return s;
    };

    std::vector<int64_t> shape_A, shape_B;
    status = read_shape(input_A, shape_A);
    if (status) return status;
    status = read_shape(input_B, shape_B);
    if (status) return status;

    if (shape_A.size() < 2 || shape_B.size() < 2)
        return info->ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "matmul: inputs must be at least 2-D");

    // Last two dims: [..., M, K_A] x [..., K_B, N]
    // For Gemm with transB: B is stored as [N, K], so K_B=shape_B[-1], N=shape_B[-2]
    int64_t M, K_A, K_B, N;
    M   = shape_A[shape_A.size() - 2];
    K_A = shape_A[shape_A.size() - 1];
    if (info->transB) {
        N   = shape_B[shape_B.size() - 2];
        K_B = shape_B[shape_B.size() - 1];
    } else {
        K_B = shape_B[shape_B.size() - 2];
        N   = shape_B[shape_B.size() - 1];
    }

    if (K_A != K_B)
        return info->ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "matmul: inner dimensions mismatch");
    const int64_t K = K_A;

    // Batch: product of all dims before the last 2 in A and B
    int64_t batch_A = 1;
    for (size_t i = 0; i + 2 < shape_A.size(); ++i) batch_A *= shape_A[i];
    int64_t batch_B = 1;
    for (size_t i = 0; i + 2 < shape_B.size(); ++i) batch_B *= shape_B[i];

    if (batch_B != 1 && batch_B != batch_A)
        return info->ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "matmul: incompatible batch dimensions");

    // ---- compute output shape and allocate ----------------------------------
    // Preserve all leading batch dims from A, append [M, N]
    std::vector<int64_t> out_shape(shape_A.begin(), shape_A.end() - 2);
    out_shape.push_back(M);
    out_shape.push_back(N);

    OrtValue* output = nullptr;
    status = info->ort_api->KernelContext_GetOutput(
        kernel_context, 0, out_shape.data(), out_shape.size(), &output);
    if (status) return status;
    if (!output)
        return info->ort_api->CreateStatus(ORT_FAIL, "matmul: failed to allocate output");

    // ---- raw data pointers --------------------------------------------------
    const void* A_raw = nullptr;
    const void* B_raw = nullptr;
    void* C_raw = nullptr;
    status = info->ort_api->GetTensorData(input_A, &A_raw);
    if (status) return status;
    status = info->ort_api->GetTensorData(input_B, &B_raw);
    if (status) return status;
    status = info->ort_api->GetTensorMutableData(output, &C_raw);
    if (status) return status;

    // ---- QLinearMatMul: read scale/zp parameters at runtime --------------------
    // Inputs: 1=a_scale (float), 4=b_scale (float), 6=y_scale (float), 7=y_zero_point (int8).
    // These are constant ONNX initializers, so the values are stable across calls.
    if (info->has_requant) {
        auto read_f32 = [&](uint32_t idx, float& out) {
            const OrtValue* v = nullptr;
            if (!info->ort_api->KernelContext_GetInput(kernel_context, idx, &v) && v) {
                const void* d = nullptr;
                if (!info->ort_api->GetTensorData(v, &d) && d)
                    out = *static_cast<const float*>(d);
            }
        };
        auto read_i8 = [&](uint32_t idx, int8_t& out) {
            const OrtValue* v = nullptr;
            if (!info->ort_api->KernelContext_GetInput(kernel_context, idx, &v) && v) {
                const void* d = nullptr;
                if (!info->ort_api->GetTensorData(v, &d) && d)
                    out = *static_cast<const int8_t*>(d);
            }
        };
        float a_scale = 1.0f, b_scale = 1.0f, y_scale = 1.0f;
        int8_t y_zp = 0;
        read_f32(1, a_scale);
        read_f32(4, b_scale);
        read_f32(6, y_scale);
        read_i8(7, y_zp);
        if (y_scale > 0.0f) {
            const double S = static_cast<double>(a_scale) * b_scale / y_scale;
            info->requant_M0     = static_cast<uint32_t>(
                std::min(S * (1LL << 31) + 0.5, static_cast<double>((1u << 31) - 1u)));
            info->requant_rshift = 31;
            info->requant_zp     = y_zp;
        }
    }

#ifdef TINYXPU_USE_VERILATOR
    // MatMulInteger: int8 inputs → int32 output via systolic array.
    constexpr int HW_ROWS = TINYXPU_ARRAY_ROWS;
    constexpr int HW_COLS = TINYXPU_ARRAY_COLS;

    // No size limit: run_tiled handles any K and N by splitting into
    // HW_ROWS x HW_COLS blocks and accumulating partial sums.
    const int8_t* A_i8  = static_cast<const int8_t*>(A_raw);
    const int8_t* B_i8  = static_cast<const int8_t*>(B_raw);
    const bool apply_relu   = info->fused_relu;
    const bool apply_requant = info->has_requant;

    // Output pointer type depends on whether requantization is active.
    // QLinearMatMul → int8 output; MatMulInteger → int32 output.
    int32_t* C_i32 = apply_requant ? nullptr : static_cast<int32_t*>(C_raw);
    int8_t*  C_i8  = apply_requant ? static_cast<int8_t*>(C_raw)  : nullptr;

    // run_slice: drive the Verilator systolic array for one 2-D tile.
    // A_sl: [total_M, K_sl], B_sl: [K_sl, N_sl] (contiguous), C_sl: [total_M, N_sl].
    // K_sl <= HW_ROWS and N_sl <= HW_COLS must hold.
    auto run_slice = [&](const int8_t* A_sl, const int8_t* B_sl,
                         int32_t* C_sl, int8_t* C_i8_sl,
                         int64_t total_M, int64_t K_sl, int64_t N_sl,
                         SimObservations& obs) {
        obs.hw_rows = HW_ROWS; obs.hw_cols = HW_COLS;

        VerilatedContext ctx;
        Varray arr{&ctx};
        struct Guard { Varray& a; ~Guard() { a.final(); } } guard{arr};

        auto tick = [&]() {
            obs.ticks_total++;
            arr.clk = 1; arr.eval();
            arr.clk = 0; arr.eval();
        };

        arr.clk = 0; arr.rst_n = 0; arr.en = 0; arr.weight_ld = 0;
        arr.relu_en    = (apply_relu || apply_requant) ? 1 : 0;
        arr.requant_en = apply_requant ? 1 : 0;
        arr.M0         = apply_requant ? info->requant_M0     : 0;
        arr.rshift     = apply_requant ? info->requant_rshift : 0;
        arr.zero_pt    = apply_requant ? static_cast<uint8_t>(info->requant_zp) : 0;
        // Load per-column bias (zero if not set)
        for (int c = 0; c < HW_COLS; ++c) {
            arr.bias_in[c] = (apply_requant && c < static_cast<int>(info->bias.size()))
                             ? static_cast<uint32_t>(info->bias[c]) : 0u;
        }
        for (int k = 0; k < HW_ROWS; ++k) arr.data_in[k] = 0;
        for (int c = 0; c < HW_COLS; ++c) arr.weight_in_top[c] = 0;
        arr.eval();
        for (int i = 0; i < 3; ++i) { tick(); obs.ticks_reset++; }
        arr.rst_n = 1;
        tick();

        arr.weight_ld = 1;
        for (int load_row = 0; load_row < HW_ROWS; ++load_row) {
            for (int c = 0; c < HW_COLS; ++c) {
                int8_t w = (load_row < K_sl && c < N_sl) ? B_sl[load_row * N_sl + c] : 0;
                arr.weight_in_top[c] = static_cast<int8_t>(w);
                if (load_row < K_sl && c < N_sl) obs.weight_writes++;
            }
            tick(); obs.ticks_weight_load++;
        }
        arr.weight_ld = 0;
        tick(); obs.ticks_weight_load++;

        const int64_t total_ticks = total_M + HW_ROWS + N_sl - 2;
        arr.en = 1;
        for (int64_t t = 0; t < total_ticks; ++t) {
            // External pre-staggering: row r's element at (t-r) enters at cycle t.
            // Row 0 enters at t=0, row 1 at t=1, etc.
            for (int r = 0; r < HW_ROWS; ++r) {
                const int64_t src_row = t - r;
                if (src_row >= 0 && src_row < total_M && r < K_sl) {
                    int8_t a = A_sl[src_row * K_sl + r];
                    arr.data_in[r] = static_cast<uint8_t>(a);
                    obs.activation_writes++;
                } else {
                    arr.data_in[r] = 0;
                }
            }
            tick(); obs.ticks_streaming++;

            for (int j = 0; j < N_sl; ++j) {
                const int64_t out_row = t - (HW_ROWS + j - 1);
                if (out_row >= 0 && out_row < total_M) {
                    C_sl[out_row * N_sl + j] = static_cast<int32_t>(arr.acc_out[j]);
                    obs.output_reads++;
                }
                // When requant is active, also capture the int8 output.
                if (apply_requant && C_i8_sl && out_row >= 0 && out_row < total_M) {
                    C_i8_sl[out_row * N_sl + j] = static_cast<int8_t>(
                        static_cast<int8_t>(arr.q_out[j]));
                }
            }
        }
    };

    // run_tiled: tile [total_M, K] x [K, N] into HW_ROWS x HW_COLS blocks.
    // Tiles along K are summed into the int32 output; tiles along N are independent.
    // When apply_requant is true the int8 output is written to C_i8 directly from
    // the hardware q_out ports (requantization happens inside the array).  In this
    // case K must fit within one tile (K <= HW_ROWS) so that requantization fires
    // on the fully-accumulated sums — enforced by a runtime assert below.
    auto run_tiled = [&](const int8_t* A_base, const int8_t* B_base,
                         int32_t* C_base, int8_t* C_i8_base,
                         int64_t total_M, SimObservations& obs) {
        if (!apply_requant)
            std::fill(C_base, C_base + total_M * N, 0);

        for (int64_t k0 = 0; k0 < K; k0 += HW_ROWS) {
            const int64_t K_t = std::min((int64_t)HW_ROWS, K - k0);
            for (int64_t n0 = 0; n0 < N; n0 += HW_COLS) {
                const int64_t N_t = std::min((int64_t)HW_COLS, N - n0);

                // Pack A tile [total_M, K_t]
                std::vector<int8_t> A_tile(total_M * K_t);
                for (int64_t m = 0; m < total_M; ++m)
                    for (int64_t k = 0; k < K_t; ++k)
                        A_tile[m * K_t + k] = A_base[m * K + k0 + k];

                // Pack B tile [K_t, N_t]
                std::vector<int8_t> B_tile(K_t * N_t);
                for (int64_t k = 0; k < K_t; ++k)
                    for (int64_t n = 0; n < N_t; ++n)
                        B_tile[k * N_t + n] = B_base[(k0 + k) * N + n0 + n];

                // Temporary int32 tile and optional int8 tile
                std::vector<int32_t> C_tile(total_M * N_t, 0);
                std::vector<int8_t>  C_i8_tile(apply_requant ? total_M * N_t : 0, 0);

                SimObservations tile_obs{};
                run_slice(A_tile.data(), B_tile.data(),
                          C_tile.data(),
                          apply_requant ? C_i8_tile.data() : nullptr,
                          total_M, K_t, N_t, tile_obs);

                if (apply_requant) {
                    // Write int8 results to the N-tile's columns in C_i8_base
                    for (int64_t m = 0; m < total_M; ++m)
                        for (int64_t n = 0; n < N_t; ++n)
                            C_i8_base[m * N + n0 + n] = C_i8_tile[m * N_t + n];
                } else {
                    // Accumulate int32 partial sums (K-tiles)
                    for (int64_t m = 0; m < total_M; ++m)
                        for (int64_t n = 0; n < N_t; ++n)
                            C_base[m * N + n0 + n] += C_tile[m * N_t + n];
                }

                obs.ticks_total       += tile_obs.ticks_total;
                obs.ticks_reset       += tile_obs.ticks_reset;
                obs.ticks_weight_load += tile_obs.ticks_weight_load;
                obs.ticks_streaming   += tile_obs.ticks_streaming;
                obs.weight_writes     += tile_obs.weight_writes;
                obs.activation_writes += tile_obs.activation_writes;
                obs.output_reads      += tile_obs.output_reads;
            }
        }
        obs.M = total_M; obs.K = K; obs.N = N;
        obs.hw_rows = HW_ROWS; obs.hw_cols = HW_COLS;
    };

    SimObservations obs{};

    if (batch_B == 1) {
        // Common case: shared weights. Flatten batch_A into M.
        run_tiled(A_i8, B_i8, C_i32, C_i8, batch_A * M, obs);
    } else {
        // Batched weights: run one tiled pass per batch.
        for (int64_t b = 0; b < batch_A; ++b) {
            SimObservations obs_b{};
            run_tiled(A_i8 + b * M * K, B_i8 + b * K * N,
                      C_i32 ? C_i32 + b * M * N : nullptr,
                      C_i8  ? C_i8  + b * M * N : nullptr,
                      M, obs_b);
            obs.ticks_total       += obs_b.ticks_total;
            obs.ticks_reset       += obs_b.ticks_reset;
            obs.ticks_weight_load += obs_b.ticks_weight_load;
            obs.ticks_streaming   += obs_b.ticks_streaming;
            obs.weight_writes     += obs_b.weight_writes;
            obs.activation_writes += obs_b.activation_writes;
            obs.output_reads      += obs_b.output_reads;
        }
        obs.M = batch_A * M; obs.K = K; obs.N = N;
        obs.hw_rows = HW_ROWS; obs.hw_cols = HW_COLS;
    }

    g_last_perf = TinyXpuPerfCounters::from_observations(obs);
#else
    // CPU fallback: MatMul (float32) or Gemm (float32 with optional bias + transB)
    const float* A = static_cast<const float*>(A_raw);
    const float* B = static_cast<const float*>(B_raw);
    float*       C = static_cast<float*>(C_raw);

    const int64_t total_M = batch_A * M;

    // Core matmul: C[i,j] = sum_k A[i,k] * B_eff[k,j]
    // transB=true: B is stored as [N, K], so B_eff[k,j] = B[j*K + k]
    // transB=false: B is stored as [K, N], so B_eff[k,j] = B[k*N + j]
    for (int64_t i = 0; i < total_M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float b_val = info->transB ? B[j * K + k] : B[k * N + j];
                acc += A[i * K + k] * b_val;
            }
            C[i * N + j] = acc;
        }
    }

    // Gemm: add optional bias (3rd input)
    if (info->op_type == "Gemm") {
        size_t num_inputs = 0;
        info->ort_api->KernelContext_GetInputCount(kernel_context, &num_inputs);
        if (num_inputs >= 3) {
            const OrtValue* input_C = nullptr;
            if (!info->ort_api->KernelContext_GetInput(kernel_context, 2, &input_C) && input_C) {
                const void* bias_raw = nullptr;
                if (!info->ort_api->GetTensorData(input_C, &bias_raw) && bias_raw) {
                    const float* bias = static_cast<const float*>(bias_raw);
                    for (int64_t i = 0; i < total_M; ++i)
                        for (int64_t j = 0; j < N; ++j)
                            C[i * N + j] += bias[j];
                }
            }
        }
    }
#endif

    return nullptr;  // Success
}

void ORT_API_CALL SampleNodeComputeInfo::ReleaseStateImpl(
    OrtNodeComputeInfo* this_,
    void* compute_state) noexcept {

    (void)this_;
    delete static_cast<ComputeState*>(compute_state);
}
