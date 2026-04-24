// Copyright (c) Sample EP Authors. Licensed under the MIT License.
// Sample ONNX Runtime Execution Provider Plugin Implementation
// Compatible with ONNX Runtime 1.22+

#include "tinyxpu_ep.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tinyxpu_driver.h"
#include "tinyxpu_perf.h"

// Platform-specific export macro
#if defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#elif defined(__APPLE__) || defined(__linux__)
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

// Global API pointers (initialized in CreateEpFactories)
static ApiPtrs g_apis;

// ============================================================================
// Exported Plugin Entry Points
// ============================================================================

extern "C" {

EXPORT_SYMBOL OrtStatus* ORT_API_CALL CreateEpFactories(
    const char* registration_name, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger, OrtEpFactory** factories,
    size_t max_factories, size_t* num_factories) noexcept {
  // Initialize global API pointers
  g_apis.Init(ort_api_base, default_logger);

  if (max_factories < 1) {
    return g_apis.ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT, "Need space for at least 1 factory");
  }

  // Create our factory
  auto* factory = new SampleEpFactory(registration_name, g_apis);
  factories[0] = factory->GetOrtFactory();
  *num_factories = 1;

  return nullptr;  // Success
}

EXPORT_SYMBOL OrtStatus* ORT_API_CALL ReleaseEpFactory(
    OrtEpFactory* factory) noexcept {
  auto* sample_factory = SampleEpFactory::FromOrt(factory);
  delete sample_factory;
  return nullptr;  // Success
}

// Returns a copy of the perf counters from the most recent MatMulInteger run.
// Python callers retrieve this via ctypes after session.run().
EXPORT_SYMBOL void tinyxpu_get_last_perf(TinyXpuPerfCounters* out) noexcept {
  if (out) *out = tinyxpu_get_last_perf_ref();
}

}  // extern "C"

// ============================================================================
// SampleEpFactory Implementation
// ============================================================================

SampleEpFactory* SampleEpFactory::FromOrt(OrtEpFactory* ort_factory) {
  return CONTAINER_OF(ort_factory, SampleEpFactory, factory_);
}

const SampleEpFactory* SampleEpFactory::FromOrt(
    const OrtEpFactory* ort_factory) {
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
  factory_.ValidateCompiledModelCompatibilityInfo =
      ValidateCompiledModelCompatibilityInfoImpl;
  factory_.CreateAllocator = CreateAllocatorImpl;
  factory_.ReleaseAllocator = ReleaseAllocatorImpl;
  factory_.CreateDataTransfer = CreateDataTransferImpl;
  factory_.IsStreamAware = IsStreamAwareImpl;
  factory_.CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}

SampleEpFactory::~SampleEpFactory() = default;

const char* ORT_API_CALL SampleEpFactory::GetNameImpl(
    const OrtEpFactory* this_) noexcept {
  return FromOrt(this_)->ep_name_.c_str();
}

const char* ORT_API_CALL SampleEpFactory::GetVendorImpl(
    const OrtEpFactory* this_) noexcept {
  (void)this_;
  return "SampleVendor";
}

uint32_t ORT_API_CALL SampleEpFactory::GetVendorIdImpl(
    const OrtEpFactory* this_) noexcept {
  (void)this_;
  return 0x1234;  // Sample vendor ID
}

const char* ORT_API_CALL SampleEpFactory::GetVersionImpl(
    const OrtEpFactory* this_) noexcept {
  (void)this_;
  return "1.0.0";
}

OrtStatus* ORT_API_CALL SampleEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_, const OrtHardwareDevice* const* devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* num_ep_devices) noexcept {
  auto* factory = FromOrt(this_);
  const auto& apis = factory->GetApis();

  *num_ep_devices = 0;

  // Look through available hardware devices and claim CPU devices
  for (size_t i = 0; i < num_devices && *num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice* hw_device = devices[i];

    // Use the API to get the device type (OrtHardwareDevice is opaque)
    OrtHardwareDeviceType device_type =
        apis.ort_api->HardwareDevice_Type(hw_device);

    // For this sample, we support CPU devices
    if (device_type == OrtHardwareDeviceType_CPU) {
      OrtEpDevice* ep_device = nullptr;
      OrtStatus* status = apis.ep_api->CreateEpDevice(this_, hw_device,
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

OrtStatus* ORT_API_CALL SampleEpFactory::CreateEpImpl(OrtEpFactory* this_,
    const OrtHardwareDevice* const* devices,
    const OrtKeyValuePairs* const* ep_metadata_pairs, size_t num_devices,
    const OrtSessionOptions* session_options, const OrtLogger* logger,
    OrtEp** ep) noexcept {
  (void)devices;
  (void)ep_metadata_pairs;
  (void)num_devices;
  (void)session_options;
  (void)logger;

  auto* factory = FromOrt(this_);
  auto* sample_ep = new SampleEp(factory);
  *ep = sample_ep->GetOrtEp();
  return nullptr;
}

void ORT_API_CALL SampleEpFactory::ReleaseEpImpl(
    OrtEpFactory* this_, OrtEp* ep) noexcept {
  (void)this_;
  auto* sample_ep = SampleEp::FromOrt(ep);
  delete sample_ep;
}

OrtStatus* ORT_API_CALL
SampleEpFactory::ValidateCompiledModelCompatibilityInfoImpl(OrtEpFactory* this_,
    const OrtHardwareDevice* const* devices, size_t num_devices,
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
    OrtEpFactory* this_, const OrtMemoryInfo* memory_info,
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
    OrtEpFactory* this_, OrtDataTransferImpl** data_transfer) noexcept {
  (void)this_;
  *data_transfer = nullptr;  // No custom data transfer needed for CPU EP
  return nullptr;
}

bool ORT_API_CALL SampleEpFactory::IsStreamAwareImpl(
    const OrtEpFactory* this_) noexcept {
  (void)this_;
  return false;
}

OrtStatus* ORT_API_CALL SampleEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_, const OrtMemoryDevice* memory_device,
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

SampleEp::SampleEp(SampleEpFactory* factory) : factory_(factory) {
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

OrtStatus* ORT_API_CALL SampleEp::GetCapabilityImpl(OrtEp* this_,
    const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) noexcept {
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
    printf("  [TinyXPU EP] Node %zu: op_type=%s\n", i,
        op_type ? op_type : "<unknown>");
    fflush(stdout);

    const bool is_matmul_integer =
        op_type && std::strcmp(op_type, "MatMulInteger") == 0;
    const bool is_qlinear_matmul =
        op_type && std::strcmp(op_type, "QLinearMatMul") == 0;
    // const bool is_relu            = op_type && std::strcmp(op_type, "Relu")
    // == 0;
    const bool is_our_op =
        is_matmul_integer;  // || is_qlinear_matmul || is_relu;
    if (!is_our_op) continue;

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
                  ONNXTensorElementDataType et =
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
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
      printf(
          "  [TinyXPU EP] Claiming op: MatMulInteger (int8, %dx%d systolic "
          "array)\n",
          TINYXPU_ARRAY_ROWS, TINYXPU_ARRAY_COLS);
    } else if (is_qlinear_matmul) {
      // QLinearMatMul: fully-integer matmul with embedded requant scales.
      // All scale/zero-point inputs (1,2,4,5,6,7) are constant initializers.
      printf(
          "  [TinyXPU EP] Claiming op: QLinearMatMul (%dx%d systolic array + "
          "requant)\n",
          TINYXPU_ARRAY_ROWS, TINYXPU_ARRAY_COLS);
    } else {
      // Relu: no type-check needed.
      printf("  [TinyXPU EP] Claiming op: Relu\n");
    }
    fflush(stdout);

    // Add each supported node as its own fused group
    const OrtNode* single_node = node;
    status = apis.ep_api->EpGraphSupportInfo_AddNodesToFuse(
        graph_support_info, &single_node, 1, nullptr);

    if (status != nullptr) {
      return status;
    }
  }

  return nullptr;  // Success
}

OrtStatus* ORT_API_CALL SampleEp::CompileImpl(OrtEp* this_,
    const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count,
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
    if (fused_nodes && fused_nodes[i]) {
      OrtStatus* s =
          apis.ort_api->Node_GetOperatorType(fused_nodes[i], &op_type_cstr);
      if (s) return s;
    }
    const std::string op_type_str = op_type_cstr ? op_type_cstr : "";

    // For Gemm: infer transB from B's stored shape vs output's N dimension.
    // B stored as [K, N] → transB=0; B stored as [N, K] → transB=1.
    bool transB = false;
    if (op_type_str == "Gemm" && fused_nodes && fused_nodes[i]) {
      // Read output shape to get N
      int64_t out_N = -1;
      {
        size_t nout = 0;
        if (!apis.ort_api->Node_GetNumOutputs(fused_nodes[i], &nout) &&
            nout >= 1) {
          std::vector<const OrtValueInfo*> ov(nout, nullptr);
          if (!apis.ort_api->Node_GetOutputs(fused_nodes[i], ov.data(), nout) &&
              ov[0]) {
            const OrtTypeInfo* ti = nullptr;
            if (!apis.ort_api->GetValueInfoTypeInfo(ov[0], &ti) && ti) {
              const OrtTensorTypeAndShapeInfo* tsi = nullptr;
              if (!apis.ort_api->CastTypeInfoToTensorInfo(ti, &tsi) && tsi) {
                size_t ndim = 0;
                if (!apis.ort_api->GetDimensionsCount(tsi, &ndim) &&
                    ndim >= 1) {
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
        if (!apis.ort_api->Node_GetNumInputs(fused_nodes[i], &nin) &&
            nin >= 2) {
          std::vector<const OrtValueInfo*> iv(nin, nullptr);
          if (!apis.ort_api->Node_GetInputs(fused_nodes[i], iv.data(), nin) &&
              iv[1]) {
            const OrtTypeInfo* ti = nullptr;
            if (!apis.ort_api->GetValueInfoTypeInfo(iv[1], &ti) && ti) {
              const OrtTensorTypeAndShapeInfo* tsi = nullptr;
              if (!apis.ort_api->CastTypeInfoToTensorInfo(ti, &tsi) && tsi) {
                size_t ndim = 0;
                if (!apis.ort_api->GetDimensionsCount(tsi, &ndim) &&
                    ndim >= 2) {
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

    auto* driver = new TinyXPUDriver(apis, op_type_str, transB);
    // supports requantization. QLinearMatMul must be handled outside this EP or
    // requant must be done in software after the array returns int32.

    node_compute_infos[i] = driver->GetOrtComputeInfo();

    // Set ep_context_nodes to nullptr since we don't support EPContext models
    if (ep_context_nodes) {
      ep_context_nodes[i] = nullptr;
    }
  }

  return nullptr;  // Success
}

void ORT_API_CALL SampleEp::ReleaseNodeComputeInfosImpl(OrtEp* this_,
    OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {
  (void)this_;

  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    auto* driver = TinyXPUDriver::FromOrt(node_compute_infos[i]);
    delete driver;
  }
}

OrtStatus* ORT_API_CALL SampleEp::GetPreferredDataLayoutImpl(
    OrtEp* this_, OrtEpDataLayout* preferred_data_layout) noexcept {
  (void)this_;
  *preferred_data_layout = OrtEpDataLayout::OrtEpDataLayout_NCHW;
  return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::ShouldConvertDataLayoutForOpImpl(OrtEp* this_,
    const char* domain, const char* op_type, OrtEpDataLayout target_data_layout,
    int* should_convert) noexcept {
  (void)this_;
  (void)domain;
  (void)op_type;
  (void)target_data_layout;
  *should_convert = -1;  // Let ORT decide
  return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::SetDynamicOptionsImpl(OrtEp* this_,
    const char* const* option_keys, const char* const* option_values,
    size_t num_options) noexcept {
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

OrtStatus* ORT_API_CALL SampleEp::EpCreateAllocatorImpl(OrtEp* this_,
    const OrtMemoryInfo* memory_info, OrtAllocator** allocator) noexcept {
  (void)this_;
  (void)memory_info;
  *allocator = nullptr;  // Use default
  return nullptr;
}

OrtStatus* ORT_API_CALL SampleEp::EpCreateSyncStreamForDeviceImpl(OrtEp* this_,
    const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream) noexcept {
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
