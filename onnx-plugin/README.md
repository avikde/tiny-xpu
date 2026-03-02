# ONNX Runtime Execution Provider Plugin

A sample implementation of an ONNX Runtime Execution Provider (EP) plugin that can be loaded dynamically at runtime. It is designed for onnxruntime 1.23+.

## Overview

This repository demonstrates how to create a custom Execution Provider plugin for ONNX Runtime 1.22+. The plugin:

- Exports the required `CreateEpFactories` and `ReleaseEpFactory` C functions
- Implements `OrtEpFactory` to create EP instances and advertise supported devices
- Implements `OrtEp` to handle node capability detection and kernel compilation
- Implements `OrtNodeComputeInfo` with `CreateState`, `Compute`, and `ReleaseState` callbacks
- Supports `Add` and `Mul` operators as a demonstration

## Installing ONNX Runtime on Linux / WSL

### Install from Pre-built Release

```bash
# Create installation directory
sudo mkdir -p /opt/onnxruntime

# Download the latest release (check https://github.com/microsoft/onnxruntime/releases)
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz

# Extract to /opt/onnxruntime
sudo tar -xzf onnxruntime-linux-x64-1.23.2.tgz -C /opt/onnxruntime --strip-components=1

# Verify installation
ls /opt/onnxruntime/include
ls /opt/onnxruntime/lib
```

### Build from source (Linux / WSL)

```bash
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.23.2
# Build, skipping tests
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
cmake --install build/Linux/RelWithDebInfo --prefix $HOME/onnxruntime_install
```

### Configure Library Path

Add the ONNX Runtime library to your library path:

```bash
# Add to your ~/.bashrc or ~/.zshrc
# echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
# built from source
echo 'export LD_LIBRARY_PATH=$HOME/onnxruntime_install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Building the Plugin

### Clone and Build

```bash
# Clone this repository
git clone https://github.com/avikde/onnx-plugin.git
cd onnx-plugin
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build . --parallel
```

After building, you'll have:

- `libsample_ep.so` - The plugin EP shared library
- `test_sample_ep` - A test application (if `BUILD_TEST_APP=ON`)

## Project Structure

```
onnx-plugin/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── include/
│   └── sample_ep.h          # EP header with class definitions
├── src/
│   └── sample_ep.cpp        # EP implementation
└── test/
    └── test_sample_ep.cpp   # Test application
```

## Plugin Architecture

### Entry Points

The plugin exports two required C functions:

```cpp
// Called by ORT to create EP factories
OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories);

// Called by ORT to release a factory
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

### OrtEpFactory

The factory creates EP instances and reports supported devices. Required callbacks:

| Callback | Purpose |
|----------|---------|
| `GetName()` | Returns EP name (e.g., "SamplePluginExecutionProvider") |
| `GetVendor()` | Returns vendor name |
| `GetVendorId()` | Returns PCI vendor ID or equivalent (1.23+) |
| `GetVersion()` | Returns semantic version string (1.23+) |
| `GetSupportedDevices()` | Enumerates hardware devices the EP can use |
| `CreateEp()` | Creates an EP instance for a session |
| `ReleaseEp()` | Releases an EP instance |

### OrtEp

The EP instance handles model execution. Required callbacks:

| Callback | Purpose |
|----------|---------|
| `GetName()` | Returns EP name (must match factory) |
| `GetCapability()` | Reports which nodes the EP can handle |
| `Compile()` | Compiles supported nodes into executable kernels |
| `ReleaseNodeComputeInfos()` | Cleans up compiled kernel info |

### OrtNodeComputeInfo

Provides the compute kernel for fused nodes:

| Callback | Purpose |
|----------|---------|
| `CreateState()` | Creates per-invocation state for the kernel |
| `Compute()` | Executes the kernel computation |
| `ReleaseState()` | Releases the state created by CreateState |

## Key Implementation Details

### Composition Pattern

The implementation uses composition rather than inheritance from `OrtEpFactory`/`OrtEp` structures:

```cpp
class SampleEpFactory {
    OrtEpFactory factory_;  // Embedded struct, returned to ORT
    std::string ep_name_;
    ApiPtrs apis_;

    // Static callbacks use CONTAINER_OF to get back to SampleEpFactory
    static SampleEpFactory* FromOrt(OrtEpFactory* ort_factory);
};
```

### noexcept Requirement

All callback functions must be declared `noexcept` to match ORT's `NO_EXCEPTION` specification:

```cpp
static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_) noexcept;
```

### API Access

ONNX Runtime APIs are accessed through function tables obtained at initialization:

```cpp
struct ApiPtrs {
    const OrtApi* ort_api;     // Main ONNX Runtime C API
    const OrtEpApi* ep_api;    // EP-specific API (GetEpApi())
};
```

## Extending the Plugin

### Adding Support for More Operators

Edit `SampleEp::GetCapabilityImpl()` in `src/sample_ep.cpp`:

```cpp
// Support additional operators
if (op_type && (std::strcmp(op_type, "Add") == 0 ||
               std::strcmp(op_type, "Mul") == 0 ||
               std::strcmp(op_type, "Sub") == 0 ||
               std::strcmp(op_type, "Div") == 0)) {
    supported_nodes.push_back(node);
}
```

Then implement the computation logic in `SampleNodeComputeInfo::ComputeImpl()`.

### Adding Hardware Device Support

To support actual hardware (GPU, NPU, etc.):

#### 1. Device Discovery

In `GetSupportedDevicesImpl()`, filter devices by hardware type:

```cpp
OrtHardwareDeviceType hw_type = apis.ort_api->HardwareDevice_Type(device);
if (hw_type == OrtHardwareDeviceType_GPU) {
    // Add this device to supported list
}
```

#### 2. Memory Management

Implement the `CreateAllocator` callback to provide device memory allocation:

```cpp
OrtStatus* CreateAllocatorImpl(OrtEpFactory* this_,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* allocator_options,
    OrtAllocator** allocator) noexcept {
    // Create and return a custom allocator for device memory
    *allocator = my_device_allocator;
    return nullptr;
}
```

#### 3. Data Transfer

Implement `CreateDataTransfer` for host-device memory copies:

```cpp
OrtStatus* CreateDataTransferImpl(OrtEpFactory* this_,
    OrtDataTransferImpl** data_transfer) noexcept {
    // Create data transfer implementation for host <-> device copies
    *data_transfer = my_data_transfer;
    return nullptr;
}
```

#### 4. Synchronous vs Asynchronous Execution

EPs can execute in two modes:

**Synchronous (stream-unaware):** The default mode. `ComputeImpl()` must block until results are ready:

```cpp
bool IsStreamAwareImpl(const OrtEpFactory* this_) noexcept {
    return false;  // Synchronous execution
}

OrtStatus* ComputeImpl(...) noexcept {
    launch_kernel(...);
    wait_for_completion();  // Must block here
    return nullptr;
}
```

**Asynchronous (stream-aware):** For hardware with command queues. Work is launched and ORT handles synchronization:

```cpp
bool IsStreamAwareImpl(const OrtEpFactory* this_) noexcept {
    return true;  // Async execution via streams
}

OrtStatus* CreateSyncStreamForDeviceImpl(OrtEpFactory* this_,
    const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* stream_options,
    OrtSyncStreamImpl** stream) noexcept {
    // Create a stream/queue handle for async execution
    *stream = my_stream_wrapper;
    return nullptr;
}

OrtStatus* ComputeImpl(...) noexcept {
    launch_kernel_async(stream, ...);  // Non-blocking
    return nullptr;  // ORT syncs the stream when needed
}
```

#### 5. Hardware Dispatch

In `ComputeImpl()`, dispatch computation to your hardware:

```cpp
OrtStatus* ComputeImpl(...) noexcept {
    // Get input/output tensors from kernel context
    // Copy data to device if needed (or use device allocator)
    // Launch hardware kernel
    // Copy results back if needed
    return nullptr;
}
```

## API Reference

- [ONNX Runtime Plugin EP Libraries Documentation](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html)
- [ONNX Runtime C API](https://onnxruntime.ai/docs/api/c/)
- [Example Plugin EP (ORT repo)](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/autoep/library)

## Troubleshooting

### "cannot open shared object file"

Add the ONNX Runtime library path:

```bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
```

### "undefined symbol" errors

Ensure your ONNX Runtime version supports the EP Plugin API (1.23+).

### Invalid conversion to 'noexcept' function pointer

All callback implementations must include `noexcept`:

```cpp
// Wrong:
static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_);

// Correct:
static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_) noexcept;
```

## Version Compatibility

The EP Plugin API evolved across ONNX Runtime versions:

| Version | Changes |
|---------|---------|
| 1.22 | Initial EP Plugin API |
| 1.23 | Added `GetVendorId()`, `GetVersion()`, `ValidateCompiledModelCompatibilityInfo()` |

## License

MIT License - see [LICENSE](LICENSE) for details.
