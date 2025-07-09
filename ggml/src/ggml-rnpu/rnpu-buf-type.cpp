#include "ggml-rnpu.h"
#include <cstdint>
#include <mutex>

#include "../ggml-backend-impl.h"

#include "./common.hpp"

// --------------------------------------------------------------------------------------
// ----------------------------- NPU buffer type interface ------------------------------
// --------------------------------------------------------------------------------------













static const ggml_backend_buffer_type_i ggml_backend_rnpu_buffer_type_interface = {
    NULL
    // /* .get_name         = */ ggml_backend_rnpu_buffer_type_name,
    // /* .alloc_buffer     = */ ggml_backend_rnpu_buffer_type_alloc_buffer,
    // /* .get_alignment    = */ ggml_backend_rnpu_buffer_type_get_alignment,
    // /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    // /* .get_alloc_size   = */ ggml_backend_rnpu_buffer_type_get_alloc_size,
    // /* .is_host          = */ ggml_backend_rnpu_buffer_type_is_host,
};



// ------------------------------------------------------------------------------------
// -------------------------- NPU buffer type external API ----------------------------
// ------------------------------------------------------------------------------------

/**
 * @brief Retrieve R-NPU buffer type for a specified device.
 *
 * This function initializes and returns the buffer type associated
 * with the given device. The buffer type of all NPUs is initialized
 * once and reused on subsequent calls. This function ensures
 * thread-safe access by using a mutex.
 *
 * @param device The device ID for retrieving buffer type.
 * @return Pointer to buffer type for a specified device, or nullptr if
 * the device ID is out of range.
 */
ggml_backend_buffer_type_t ggml_backend_rnpu_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_rnpu_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_rnpu_buffer_types[GGML_RNPU_MAX_DEVICES];

    static bool ggml_backend_rnpu_buffer_type_initialized = false;

    if (!ggml_backend_rnpu_buffer_type_initialized) {
        for (int32_t i = 0; i < ggml_rnpu_info().device_count; ++i) {
            ggml_backend_rnpu_buffer_types[i] = {
                /* .iface    = */ ggml_backend_rnpu_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_rnpu_reg(), i),
                /* .context  = */ new ggml_backend_rnpu_buffer_type_context{i, GGML_RNPU_NAME + std::to_string(i)},
            };
        }
        ggml_backend_rnpu_buffer_type_initialized = true;
    }

    return &ggml_backend_rnpu_buffer_types[device];
}