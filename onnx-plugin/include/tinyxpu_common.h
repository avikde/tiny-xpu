#pragma once
// Common utilities shared across the TinyXPU ONNX EP plugin.

#include <onnxruntime_c_api.h>
#include <cstddef>

// Helper macro to get containing object from member pointer (like container_of in Linux kernel)
#define CONTAINER_OF(ptr, type, member) \
    reinterpret_cast<type*>(reinterpret_cast<char*>(ptr) - offsetof(type, member))

#define CONTAINER_OF_CONST(ptr, type, member) \
    reinterpret_cast<const type*>(reinterpret_cast<const char*>(ptr) - offsetof(type, member))

// ============================================================================
// Utility structure to hold API pointers
// ============================================================================
struct ApiPtrs {
    const OrtApi* ort_api = nullptr;
    const OrtEpApi* ep_api = nullptr;
    const OrtLogger* logger = nullptr;

    void Init(const OrtApiBase* api_base, const OrtLogger* log) {
        ort_api = api_base->GetApi(ORT_API_VERSION);
        ep_api = ort_api->GetEpApi();
        logger = log;
    }
};
