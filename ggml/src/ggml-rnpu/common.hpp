#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../ggml-backend-impl.h"

#define GGML_RNPU_MAX_DEVICES 1
#define GGML_RNPU_MAX_STREAMS 1
#define GGML_RNPU_NAME "RNPU"

#define MATRIX_ROW_PADDING 512

#define NPU_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_npu_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define NPU_CHECK(err) NPU_CHECK_GEN(err, npuSuccess, npuGetErrorString)




#ifdef  __cplusplus
extern "C" {
#endif

// -------------------------------------------------------------------
// ------------------------- NPU device info -------------------------
// -------------------------------------------------------------------

/**
 * @brief Contains information about R-NPU devices
 */
struct ggml_rnpu_device_info {
    /**
     * @brief Number of R-NPU devices available
     */
    int32_t device_count;

    /**
     * @brief Information about a single R-NPU device
     */
    struct rnpu_device_info {
        // TODO: confirm/add device info (?)
        int     cc;                 // Compute capability.
        // bool    vmm;                // Virtual memory support.
        // size_t  vmm_granularity;    // Granularity of virtual memory.
        // size_t  total_vram;         // Total virtual RAM available on the device.
    };

    rnpu_device_info devices[GGML_RNPU_MAX_DEVICES] =
        {}; // Array of R-NPU device information.
};



// -------------------------------------------------------------------
// --------------------------- NPU context ---------------------------
// -------------------------------------------------------------------

/**
 * @brief Contains context information in backend registry
 */
struct ggml_backend_rnpu_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

/**
 * @brief Contains context information in a device.
 */
struct ggml_backend_rnpu_device_context {
    int device;
    std::string name;
    std::string description;
};

/**
 * @brief Contains context information in a device backend.
 */
struct ggml_backend_rnpu_context {
    int device;
    std::string name;
    // TODO: NPU Event
    // npuEvent_t copy_event = nullptr;

    // TODO: NPU Stream
    // npuStream_t streams[GGML_RNPU_MAX_STREAMS] = {nullptr};

    // TODO: more extended NPU context (?)

    explicit ggml_backend_rnpu_context(int device):
        device(device),
        name(GGML_RNPU_NAME + std::to_string(device)) { }

    ~ggml_backend_rnpu_context() {
        // TODO: Release resources
        // if (copy_event != nullptr) {
        //     NPU_CHECK(npuEventDestory(copy_event))
        // }
        // for (int j = 0; j < GGML_RNPU_MAX_STREAMS; ++j) {
        //     if (streams[j] != nullptr) {
        //         NPU_CHECK(npuStreamDestroy(streams[j]))
        //     }
        // }
    }

    // TODO: stream API
    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    // npuStream stream(int stream) {
    //     if (streams[stream] == nullptr) {
    //         ggml_rnpu_set_device(device);
    //         NPU_CHECK(npuCreateStream(&streams[stream]))
    //     }
    //     return streams[stream];
    // }

    // TODO: stream API
    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    // npuStream stream() { return stream(0) }

    // TODO: More API for extended NPU resources (?)
};

/**
 * @brief Contains context information in a device backend buffer type.
 */
struct ggml_backend_rnpu_buffer_type_context {
    int32_t device; // Device ID associated with the buffer context.
    std::string name; // Name associated with the buffer context.
};

/**
 * @brief Contains context information of a device backend buffer.
 *
 * This structure holds information about a R-NPU buffer, including the device
 * ID, device buffer pointer.
 */
struct ggml_backend_rnpu_buffer_context {
    int32_t device; // The device ID associated with buffer
    void * dev_ptr =
            nullptr; // Pointer to device memory allocated for this buffer.

    ggml_backend_rnpu_buffer_context(int32_t device, void * dev_ptr)
      : device(device),
        dev_ptr(dev_ptr) {}

    ~ggml_backend_rnpu_buffer_context() {
        // TODO: runtime free API
        // NPU_CHECK(cudartFree(dev_ptr))
    }
};



// -------------------------------------------------------------------
// --------------------- NPU common internal API ---------------------
// -------------------------------------------------------------------

[[noreturn]] void ggml_npu_error(
                    const char * stmt, const char * func,
                    const char * file, int line, const char * msg);

void ggml_rnpu_set_device(int32_t device);

int32_t ggml_rnpu_get_device();

const ggml_rnpu_device_info& ggml_rnpu_info();

bool ggml_rnpu_compute_forward(ggml_backend_rnpu_context& ctx,
                               struct ggml_tensor* dst);

// rnpu-buf-type.cpp
bool ggml_backend_buft_is_rnpu(ggml_backend_buffer_type_t buft);

// rnpu-buf.cpp
bool ggml_backend_buffer_is_rnpu(ggml_backend_buffer_t buffer);

#ifdef  __cplusplus
}
#endif
