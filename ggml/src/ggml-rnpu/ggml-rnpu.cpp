#include "ggml-rnpu.h"
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "../ggml-impl.h"
#include "../ggml-backend-impl.h"
#include "./common.hpp"
#include "ggml.h"

// --------------------------------------------------------------------------------
// ---------------------------- NPU common internal API ---------------------------
// --------------------------------------------------------------------------------

/**
 * @brief Handles R-NPU errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_npu_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int32_t id = -1;
    // TODO: get device API
    // npurtGetDevice(&id);

    GGML_LOG_ERROR("R-NPU error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func,
            file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("R-NPU error");
}

/**
 * @brief Retrieve the current device ID
 *
 * @return The current device ID
 */
int32_t ggml_rnpu_get_device() {
    int32_t id;
    // TODO: get device API
    // NPU_CHECK(npurtGetDevice(&id));
    return id;
}

/**
 * @brief Set the devices to be used by R-NPU.
 *
 * @param device The device ID to set.
 */
void ggml_rnpu_set_device(const int32_t device) {
    // TODO: set device API
    // NPU_CHECK(npurtSetDevice(device))
}

/**
 * @brief Initialize R-NPU device information
 *
 * This function initializes R-NPU device information by obtaining the device
 * count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_rnpu_device_info ggml_rnpu_init() {
    ggml_rnpu_device_info info = {};

    // TODO: device count API
    // npuError err = npurtGetDeviceCount((uint32_t *)&info.device_count);
    // if (err != NPU_SUCCESS) {
    //     GGML_LOG_ERROR("%s: failed to initialize R-NPU: %s\n",
    //                     __func__, npuGetErrorString());
    //     return info;
    // }

    GGML_ASSERT(info.device_count <= GGML_RNPU_MAX_DEVICES);

    return info;
}

/**
 * @brief Retrieve R-NPU device information.
 *
 * This function returns a reference to a structure containing the R-NPU device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A refernce to the structure containing the device information.
 */
const ggml_rnpu_device_info& ggml_rnpu_info() {
    static ggml_rnpu_device_info info = ggml_rnpu_init();
    return info;
}
