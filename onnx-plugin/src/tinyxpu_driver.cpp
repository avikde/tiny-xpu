#include "tinyxpu_driver.h"
#include "tinyxpu_perf.h"

#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef TINYXPU_USE_VERILATOR
#include "Varray.h"
#include "verilated.h"
// Required when Verilator is built with SystemC support (e.g. Homebrew on macOS)
double sc_time_stamp() { return 0; }
#endif

// Simple compute state - just stores a flag
struct ComputeState {
    bool initialized = true;
};

TinyXPUDriver* TinyXPUDriver::FromOrt(OrtNodeComputeInfo* ort_info) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#endif
    return CONTAINER_OF(ort_info, TinyXPUDriver, compute_info_);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

TinyXPUDriver::TinyXPUDriver(const ApiPtrs& apis, std::string op_type_str,
                             bool transB_flag, bool /* fused_relu_flag */)
    : ort_api(apis.ort_api), ep_api(apis.ep_api),
      op_type(std::move(op_type_str)), transB(transB_flag) {
    // NOTE: fused_relu, has_requant, requant_M0, requant_rshift, requant_zp, bias removed.

    // Zero-initialize the OrtNodeComputeInfo struct
    std::memset(&compute_info_, 0, sizeof(compute_info_));

    compute_info_.ort_version_supported = ORT_API_VERSION;

    compute_info_.CreateState = CreateStateImpl;
    compute_info_.Compute = ComputeImpl;
    compute_info_.ReleaseState = ReleaseStateImpl;
}

OrtStatus* ORT_API_CALL TinyXPUDriver::CreateStateImpl(
    OrtNodeComputeInfo* this_,
    OrtNodeComputeContext* compute_context,
    void** compute_state) noexcept {

    (void)this_;
    (void)compute_context;

    // Create compute state
    *compute_state = new ComputeState();
    return nullptr;
}

OrtStatus* ORT_API_CALL TinyXPUDriver::ComputeImpl(
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
        s = info->ort_api->GetDimensionsCount(si, &ndim);
        if (s) { info->ort_api->ReleaseTensorTypeAndShapeInfo(si); return s; }
        std::vector<int64_t> shape(ndim);
        s = info->ort_api->GetDimensions(si, shape.data(), ndim);
        info->ort_api->ReleaseTensorTypeAndShapeInfo(si);
        if (s) return s;

        OrtValue* output = nullptr;
        s = info->ort_api->KernelContext_GetOutput(kernel_context, 0, shape.data(), shape.size(), &output);
        if (s) return s;

        const void* in_raw = nullptr;
        void* out_raw = nullptr;
        s = info->ort_api->GetTensorData(input, &in_raw);
        if (s) return s;
        s = info->ort_api->GetTensorMutableData(output, &out_raw);
        if (s) return s;

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

    // ---- QLinearMatMul scale/zp reading removed ------------------------------
    // QLinearMatMul still reaches ComputeImpl but must be handled in software after
    // the Verilator array returns int32 results, or handled outside this EP.
    // TODO: QLinearMatMul should be rejected in GetCapabilityImpl or CompileImpl
    // so it falls through to a different EP.

#ifdef TINYXPU_USE_VERILATOR
    // MatMulInteger: int8 inputs -> int32 output via systolic array.
    constexpr int HW_ROWS = TINYXPU_ARRAY_ROWS;
    constexpr int HW_COLS = TINYXPU_ARRAY_COLS;

    // No size limit: run_tiled handles any K and N by splitting into
    // HW_ROWS x HW_COLS blocks and accumulating partial sums.
    const int8_t* A_i8  = static_cast<const int8_t*>(A_raw);
    const int8_t* B_i8  = static_cast<const int8_t*>(B_raw);
    // QLinearMatMul and Relu must be handled in software or outside this EP.
    int32_t* C_i32 = static_cast<int32_t*>(C_raw);

    // run_slice: drive the Verilator systolic array for one 2-D tile.
    // A_sl: [total_M, K_sl], B_sl: [K_sl, N_sl] (contiguous), C_sl: [total_M, N_sl].
    // K_sl <= HW_ROWS and N_sl <= HW_COLS must hold.
    // Requantization (if needed) must happen in software after the array returns.
    auto run_slice = [&](const int8_t* A_sl, const int8_t* B_sl,
                         int32_t* C_sl, int8_t* /* C_i8_sl */,
                         int64_t total_M, int64_t K_sl, int64_t N_sl,
                         SimObservations& obs) {
        (void)C_sl;  // silence unused param warning
        obs.hw_rows = HW_ROWS; obs.hw_cols = HW_COLS;

        VerilatedContext ctx;
        Varray arr{&ctx};
        struct Guard { Varray& a; ~Guard() { a.final(); } } guard{arr};

        auto tick = [&]() {
            obs.ticks_total++;
            arr.clk = 1; arr.eval();
            arr.clk = 0; arr.eval();
        };

        arr.clk = 0; arr.rst_n = 0;
        // Initialization
        for (int c = 0; c < HW_COLS; ++c) arr.weight_ld[c] = 0;
        // Load per-column bias via acc_in_top before weight_ld if needed.
        for (int k = 0; k < HW_ROWS; ++k) arr.data_in_left[k] = 0;
        // input when weight_ld[c]=1. Weights flow through the accumulator cascade.
        for (int c = 0; c < HW_COLS; ++c) arr.acc_in_top[c] = 0;
        arr.eval();

        tick();
        obs.ticks_reset++;
        arr.rst_n = 1;
        tick();

        for (int c = 0; c < HW_COLS; ++c) arr.weight_ld[c] = 1;
        for (int load_row = 0; load_row < HW_ROWS; ++load_row) {
            for (int c = 0; c < HW_COLS; ++c) {
                int8_t w = (load_row < K_sl && c < N_sl) ? B_sl[load_row * N_sl + c] : 0;
                // Weights loaded through acc_in_top; pe.sv extracts weight_r from acc_in on weight_ld.
                arr.acc_in_top[c] = static_cast<int8_t>(w);
                if (load_row < K_sl && c < N_sl) obs.weight_writes++;
            }
            tick(); obs.ticks_weight_load++;
        }
        for (int c = 0; c < HW_COLS; ++c) arr.weight_ld[c] = 0;
        tick(); obs.ticks_weight_load++;

        const int64_t total_ticks = total_M + HW_ROWS + N_sl - 2;
        for (int64_t t = 0; t < total_ticks; ++t) {
            // External pre-staggering: row r's element at (t-r) enters at cycle t.
            // Row 0 enters at t=0, row 1 at t=1, etc.
            for (int r = 0; r < HW_ROWS; ++r) {
                const int64_t src_row = t - r;
                if (src_row >= 0 && src_row < total_M && r < K_sl) {
                    int8_t a = A_sl[src_row * K_sl + r];
                    arr.data_in_left[r] = static_cast<uint8_t>(a);
                    obs.activation_writes++;
                } else {
                    arr.data_in_left[r] = 0;
                }
            }
            tick(); obs.ticks_streaming++;

            // NOTE: acc_out_bottom replaced acc_out; q_out removed (no requant in hardware).
            for (int j = 0; j < N_sl; ++j) {
                const int64_t out_row = t - (HW_ROWS + j - 1);
                if (out_row >= 0 && out_row < total_M) {
                    C_sl[out_row * N_sl + j] = static_cast<int32_t>(arr.acc_out_bottom[j]);
                    obs.output_reads++;
                }
            }
        }
    };

    // FIXME: Tiling should be in hardware
    // run_tiled: tile [total_M, K] x [K, N] into HW_ROWS x HW_COLS blocks.
    // Tiles along K are summed into the int32 output; tiles along N are independent.
    // (if needed for QLinearMatMul) must happen in software after the array returns.
    auto run_tiled = [&](const int8_t* A_base, const int8_t* B_base,
                         int32_t* C_base, int8_t* /* C_i8_base */,
                         int64_t total_M, SimObservations& obs) {
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

                // Temporary int32 tile (accumulates partial sums across K tiles)
                std::vector<int32_t> C_tile(total_M * N_t, 0);

                SimObservations tile_obs{};
                run_slice(A_tile.data(), B_tile.data(),
                          C_tile.data(), nullptr,
                          total_M, K_t, N_t, tile_obs);

                // Accumulate int32 partial sums (K-tiles)
                for (int64_t m = 0; m < total_M; ++m)
                    for (int64_t n = 0; n < N_t; ++n)
                        C_base[m * N + n0 + n] += C_tile[m * N_t + n];

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
        run_tiled(A_i8, B_i8, C_i32, nullptr, batch_A * M, obs);
    } else {
        // Batched weights: run one tiled pass per batch.
        for (int64_t b = 0; b < batch_A; ++b) {
            SimObservations obs_b{};
            run_tiled(A_i8 + b * M * K, B_i8 + b * K * N,
                      C_i32 + b * M * N, nullptr,
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

    tinyxpu_set_last_perf(TinyXpuPerfCounters::from_observations(obs));
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

void ORT_API_CALL TinyXPUDriver::ReleaseStateImpl(
    OrtNodeComputeInfo* this_,
    void* compute_state) noexcept {

    (void)this_;
    delete static_cast<ComputeState*>(compute_state);
}
