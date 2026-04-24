#pragma once
// TinyXPU per-node execution driver.
// Implements OrtNodeComputeInfo by driving the Verilator systolic array
// or a CPU fallback for each fused graph node assigned to the EP.

#include <onnxruntime_c_api.h>

#include <string>

#include "tinyxpu_common.h"

// ============================================================================
// TinyXPUDriver - Implements computation for fused nodes
// Uses composition to wrap OrtNodeComputeInfo
// ============================================================================
class TinyXPUDriver {
 public:
  TinyXPUDriver(const ApiPtrs& apis, std::string op_type_str,
      bool transB_flag = false, bool fused_relu_flag = false);

  OrtNodeComputeInfo* GetOrtComputeInfo() { return &compute_info_; }

  static TinyXPUDriver* FromOrt(OrtNodeComputeInfo* ort_info);

  const OrtApi* ort_api;
  const OrtEpApi* ep_api;
  // TODO: op_type should be narrowed to only MatMulInteger / MatMul / Gemm.
  // QLinearMatMul and Relu are no longer supported by this EP.
  std::string op_type;  // "MatMulInteger", "MatMul", "Gemm"
  bool transB;          // Gemm only: inferred from B's shape in CompileImpl
  // NOTE: fused_relu, has_requant, requant_* fields removed — hardware no
  // longer supports ReLU or requantization. QLinearMatMul must be handled
  // externally.

 private:
  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_,
      OrtNodeComputeContext* compute_context, void** compute_state) noexcept;

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_,
      void* compute_state, OrtKernelContext* kernel_context) noexcept;

  static void ORT_API_CALL ReleaseStateImpl(
      OrtNodeComputeInfo* this_, void* compute_state) noexcept;

  OrtNodeComputeInfo compute_info_;  // The actual OrtNodeComputeInfo struct
};
