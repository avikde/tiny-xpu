// Copyright (c) Sample EP Authors. Licensed under the MIT License.
// Sample ONNX Runtime Execution Provider Plugin
//
// This file must only include onnxruntime_c_api.h - the EP API is included from
// there.

#pragma once

#include <onnxruntime_c_api.h>

#include <string>

#include "tinyxpu_common.h"

// Forward declarations
class SampleEpFactory;
class SampleEp;

// ============================================================================
// SampleEpFactory - Creates EP instances
// Uses composition to wrap OrtEpFactory
// ============================================================================
class SampleEpFactory {
 public:
  SampleEpFactory(const char* name, const ApiPtrs& apis);
  ~SampleEpFactory();

  // Get the OrtEpFactory struct to return to ORT
  OrtEpFactory* GetOrtFactory() { return &factory_; }

  const ApiPtrs& GetApis() const { return apis_; }
  const std::string& GetEpName() const { return ep_name_; }

  // Helper to get SampleEpFactory from OrtEpFactory pointer
  static SampleEpFactory* FromOrt(OrtEpFactory* ort_factory);
  static const SampleEpFactory* FromOrt(const OrtEpFactory* ort_factory);

 private:
  // OrtEpFactory callback implementations (1.23 API)
  static const char* ORT_API_CALL GetNameImpl(
      const OrtEpFactory* this_) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(
      const OrtEpFactory* this_) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(
      const OrtEpFactory* this_) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(
      const OrtEpFactory* this_) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_,
      const OrtHardwareDevice* const* devices, size_t num_devices,
      OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_,
      const OrtHardwareDevice* const* devices,
      const OrtKeyValuePairs* const* ep_metadata_pairs, size_t num_devices,
      const OrtSessionOptions* session_options, const OrtLogger* logger,
      OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(
      OrtEpFactory* this_, OrtEp* ep) noexcept;

  // 1.23 optional callback stubs
  static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      OrtEpFactory* this_, const OrtHardwareDevice* const* devices,
      size_t num_devices, const char* compatibility_info,
      OrtCompiledModelCompatibility* model_compatibility) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_,
      const OrtMemoryInfo* memory_info,
      const OrtKeyValuePairs* allocator_options,
      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* this_, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_, OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(
      const OrtEpFactory* this_) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_, const OrtMemoryDevice* memory_device,
      const OrtKeyValuePairs* stream_options,
      OrtSyncStreamImpl** stream) noexcept;

  OrtEpFactory factory_;  // The actual OrtEpFactory struct
  std::string ep_name_;
  ApiPtrs apis_;
};

// ============================================================================
// SampleEp - The actual execution provider instance
// Uses composition to wrap OrtEp
// ============================================================================
class SampleEp {
 public:
  SampleEp(SampleEpFactory* factory);
  ~SampleEp();

  // Get the OrtEp struct to return to ORT
  OrtEp* GetOrtEp() { return &ep_; }

  SampleEpFactory* GetFactory() const { return factory_; }
  const ApiPtrs& GetApis() const { return factory_->GetApis(); }

  // Helper to get SampleEp from OrtEp pointer
  static SampleEp* FromOrt(OrtEp* ort_ep);
  static const SampleEp* FromOrt(const OrtEp* ort_ep);

 private:
  // OrtEp callback implementations (1.23 API)
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_,
      const OrtGraph* graph,
      OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_,
      const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_,
      OrtNodeComputeInfo** node_compute_infos,
      size_t num_node_compute_infos) noexcept;

  // 1.23 optional callback stubs for OrtEp
  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(
      OrtEp* this_, OrtEpDataLayout* preferred_data_layout) noexcept;

  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(OrtEp* this_,
      const char* domain, const char* op_type,
      OrtEpDataLayout target_data_layout, int* should_convert) noexcept;

  static OrtStatus* ORT_API_CALL SetDynamicOptionsImpl(OrtEp* this_,
      const char* const* option_keys, const char* const* option_values,
      size_t num_options) noexcept;

  static OrtStatus* ORT_API_CALL OnRunStartImpl(
      OrtEp* this_, const OrtRunOptions* run_options) noexcept;

  static OrtStatus* ORT_API_CALL OnRunEndImpl(OrtEp* this_,
      const OrtRunOptions* run_options, bool sync_stream) noexcept;

  static OrtStatus* ORT_API_CALL EpCreateAllocatorImpl(OrtEp* this_,
      const OrtMemoryInfo* memory_info, OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL EpCreateSyncStreamForDeviceImpl(OrtEp* this_,
      const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(
      OrtEp* this_, const OrtGraph* graph) noexcept;

  OrtEp ep_;  // The actual OrtEp struct
  SampleEpFactory* factory_;
  // NOTE: session_logger_ removed — was stored but never used.
};
