#include "ggml-rnpu.h"
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "../ggml-impl.h"
#include "../ggml-backend-impl.h"
#include "./common.hpp"
#include "./ops.hpp"
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
    int32_t id = 0;
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
    // FIXME: delete in the future (if support multi-device)
    GGML_UNUSED(device);

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

    // FIXME: delete in the futre (if support multi-device)
    info.device_count = 1;
    info.devices[0].cc = 0;

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

/**
 * @brief Computes the forward operation for a given tensor using R-NPU
 * operations.
 *
 * This function selects the appropriate CANN operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The R-NPU context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @return true if the computation was successful; false otherwise.
 */
bool ggml_rnpu_compute_forward(ggml_backend_rnpu_context& ctx,
                               struct ggml_tensor* dst) {
    switch (dst->op) {
        case GGML_OP_GET_ROWS:
            ggml_rnpu_get_rows(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_rnpu_op_add(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_rnpu_op_mul(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_SILU:
                    ggml_rnpu_silu(ctx, dst);
                    break;
                // support remaining unary op
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
            ggml_rnpu_mul_mat(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_rnpu_rope(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_rnpu_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_rnpu_cont(ctx, dst);
            break;
        case GGML_OP_RESHAPE:
            ggml_rnpu_reshape(ctx, dst);
            break;
        case GGML_OP_VIEW:
            ggml_rnpu_view(ctx, dst);
            break;
        case GGML_OP_PERMUTE:
            ggml_rnpu_permute(ctx, dst);
            break;
        case GGML_OP_TRANSPOSE:
            ggml_rnpu_transpose(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_rnpu_rms_norm(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_rnpu_softmax(ctx, dst);
            break;
        // TODO: support remainig op
        default:
            return false;
    }

    return true;
}
