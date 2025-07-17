#include "ggml-backend.h"
#include "ggml-rnpu.h"
// #include "ggml-backend.h"
#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"

#include "./common.hpp"
#include "ggml.h"
#include <cstdint>

// --------------------------------------------------------------------------------
// --------------------------- NPU device interface -------------------------------
// --------------------------------------------------------------------------------

/**
 * @brief Retrieve device name from the pointer to a device.
 *
 * @param dev Pointer to the device.
 * @return Pointer to C-style string of device name.
 */
static const char * ggml_backend_rnpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rnpu_device_context * ctx = (ggml_backend_rnpu_device_context *)dev->context;
    return ctx->name.c_str();
}

/**
 * @brief Retrieve device description from the pointer to a device.
 *
 * @param dev Pointer to the device.
 * @return Pointer to C-style string of device description.
 */
static const char * ggml_backend_rnpu_device_get_description(
                            ggml_backend_dev_t dev) {
    ggml_backend_rnpu_device_context * ctx = (ggml_backend_rnpu_device_context *)dev->context;
    return ctx->description.c_str();
}

/**
 * @brief Retrieve device memory information from the pointer to a device.
 *
 * @param dev Pointer to the device.
 * @param free The free memory size in bytes on the device to be set.
 * @param total The total memory size in bytes on the device to be set.
 */
static void ggml_backend_rnpu_device_get_memory(
                ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rnpu_device_context * ctx = (ggml_backend_rnpu_device_context *)dev->context;
    ggml_backend_rnpu_get_device_memory(ctx->device, free, total);
}

/**
 * @brief Retrieve device type from the pointer to a device.
 *
 * @param dev Pointer to the device.
 * @return Always returns GPU_TYPE. R-NPU is a heterogeneous computing
 * device that has individual memory from CPU.
 */
static enum ggml_backend_dev_type ggml_backend_rnpu_device_get_type(
                                    ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

/**
 * @brief Retrieve device properties from the pointer to a device.
 *
 * @param dev Pointer to the device.
 * @param props Pointer to the device properties to be set
 */
static void ggml_backend_rnpu_device_get_props(
                ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rnpu_device_get_name(dev);
    props->description = ggml_backend_rnpu_device_get_description(dev);
    props->type        = ggml_backend_rnpu_device_get_type(dev);
    ggml_backend_rnpu_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_RNPU_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ false, // TODO: support async op in the future
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false, // TODO: support evnets in the future
    };
}

/**
 * @brief Initialize a new backend context for a specified NPU device.
 *
 * @param dev Pointer to the device.
 * @param params Pointer to the C-style string of params. (Unused)
 * @return Pointer to the initialized backend.
 */
static ggml_backend_t ggml_backend_rnpu_device_init_backend(
                        ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_rnpu_device_context * ctx = (ggml_backend_rnpu_device_context *)dev->context;
    return ggml_backend_rnpu_init(ctx->device);
}

/**
 * @brief Retrieve device buffer type context from the pointer to a device.
 *
 * @param dev Pointer to the device.
 * @return Pointer to the retrieved buffer type.
 */
static ggml_backend_buffer_type_t ggml_backend_rnpu_device_get_buffer_type(
                                    ggml_backend_dev_t dev) {
    ggml_backend_rnpu_device_context * ctx = (ggml_backend_rnpu_device_context *)dev->context;
    return ggml_backend_rnpu_buffer_type(ctx->device);
}

/**
 * @brief Retrieve R-NPU host buffer type context.
 *
 * @param dev Pointer to the device. (Unused)
 * @return Pointer to the retrieved buffer type. All R-NPU devices has
 * the same one host buffer type.
 */
static ggml_backend_buffer_type_t ggml_backend_rnpu_device_get_host_buffer_type(
                                    ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_rnpu_host_buffer_type();
}

/**
 * @brief Check whether the op of tensor is supported by a R-NPU device.
 *
 * @param dev Pointer to a R-NPU device.
 * @param op Pointer to tensor to be check.
 * @return Return 'true' if R-NPU device supports the tensor op.
 */
static bool ggml_backend_rnpu_device_supports_op(
                ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_rnpu_device_context * dev_ctx = (ggml_backend_rnpu_device_context *) dev->context;
    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_rnpu(op->src[i]->buffer->buft)) {
            ggml_backend_rnpu_buffer_type_context * buft_ctx = (ggml_backend_rnpu_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                // TODO: supports more unary op in the future
                // case GGML_UNARY_OP_ABS:
                // case GGML_UNARY_OP_NEG:
                // case GGML_UNARY_OP_GELU:
                // case GGML_UNARY_OP_RELU:
                // case GGML_UNARY_OP_SIGMOID:
                // case GGML_UNARY_OP_HARDSIGMOID:
                // case GGML_UNARY_OP_HARDSWISH:
                // case GGML_UNARY_OP_GELU_QUICK:
                // case GGML_UNARY_OP_TANH:
                // case GGML_UNARY_OP_EXP:
                // case GGML_UNARY_OP_ELU:
                // case GGML_UNARY_OP_SGN:
                // case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_SILU:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT_ID:
            return false;
        case GGML_OP_MUL_MAT:
            // NOTE: We only support e4m3q-f32 and bf16-f32 matmul now
            switch (op->src[0]->type) {
                case GGML_TYPE_BF16:
                    return true;
                case GGML_TYPE_E4M3_Q:
                    return ggml_is_contiguous(op->src[0]) &&
                            ggml_is_contiguous(op->src[1]);
                default:
                    return false;
            }
        case GGML_OP_GET_ROWS:
            switch (op->src[0]->type) {
                // NOTE: We only support get rows from fp32/bf16 tensor
                case GGML_TYPE_F32:
                case GGML_TYPE_BF16:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_ROPE: {
            const float ext_factor = ((const float *)op->op_params)[7];
            // TODO: support n_dims <= ne0
            if (op->src[0]->ne[0] != op->op_params[1]) {
                return false;
            }
            // TODO: support ext_factor != 0
            if (ext_factor != 0) {
                return false;
            }
            const int mode = ((const int32_t *)op->op_params)[2];
            // NOTE: We only support NEOX rope and contiguous input.
            if ((mode & GGML_ROPE_TYPE_MROPE) || (mode & GGML_ROPE_TYPE_VISION)) {
                return false;
            }
            if (!(mode & GGML_ROPE_TYPE_NEOX) || !ggml_is_contiguous(op->src[0])) {
                return false;
            }
            return true;
        }
        case GGML_OP_CPY: {
            // NOTE: We only support copy from contiguous F32 to BF16
            return (op->src[0]->type == GGML_TYPE_F32) &&
                   (op->type == GGML_TYPE_BF16);
        }
        case GGML_OP_CONT: {
            // NOTE: We only support copy from FP32 to FP32
            return (op->src[0]->type == GGML_TYPE_F32) &&
                   (op->type == GGML_TYPE_F32);
        }
        // TODO: support implicit DUP op (CPY and CONT already use DUP)
        case GGML_OP_DUP:
            return false;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return true;
        // TODO: supports more op in the future
        // case GGML_OP_SUM:
        // case GGML_OP_IM2COL:
        // case GGML_OP_CONCAT:
        // case GGML_OP_REPEAT:
        // case GGML_OP_NONE:
        // case GGML_OP_NORM:
        // case GGML_OP_ADD1:
        // case GGML_OP_SUB:
        // case GGML_OP_DIV:
        // case GGML_OP_SCALE:
        // case GGML_OP_SQR:
        // case GGML_OP_SQRT:
        // case GGML_OP_CLAMP:
        // case GGML_OP_DIAG_MASK_INF:
        // case GGML_OP_SUM_ROWS:
        // case GGML_OP_ARGSORT:
        // case GGML_OP_ACC:
        // case GGML_OP_GROUP_NORM:
        // case GGML_OP_PAD:
        // case GGML_OP_ARANGE:
        // case GGML_OP_TIMESTEP_EMBEDDING:
        // case GGML_OP_LEAKY_RELU:
        // case GGML_OP_ARGMAX:
        // case GGML_OP_COS:
        // case GGML_OP_SIN:
        // case GGML_OP_CONV_TRANSPOSE_1D:
        // case GGML_OP_LOG:
        // case GGML_OP_MEAN:
        // case GGML_OP_PAD_REFLECT_1D:
        // case GGML_OP_COUNT_EQUAL:
        default:
            return false;
    }
    GGML_UNUSED(dev);
    return false;
}

/**
 * @brief Check whether the buffer type is supported by a R-NPU device.
 *
 * @param dev Pointer to a R-NPU device.
 * @param buft Pointer to tensor to be check.
 * @return Return 'true' if R-NPU device supports the buffer type.
 */
static bool ggml_backend_rnpu_device_supports_buft(
                ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_rnpu(buft) && (buft->device == dev);
}

/**
 * @brief Determines if a tensor operation should be offloaded to the R-NPU
 * backend.
 *
 * This function returns true if the second dimension (ne[1]) of the given
 * tensor is greater than or equal to the minimum batch size and the operation
 * is not GGML_OP_GET_ROWS.
 *
 * @param dev Pointer to the R-NPU device.
 * @param op Pointer to the tensor to be check.
 * @return {text}
 * @return {text}
 */
static bool ggml_backend_rnpu_device_offload_op(
                ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;
    GGML_UNUSED(dev);

    return (op->ne[1] >= min_batch_size) && (op->op != GGML_OP_GET_ROWS);
}

static ggml_backend_event_t ggml_backend_rnpu_device_event_new(
                                ggml_backend_dev_t dev) {
    // TODO: support NPU Event in the future
    GGML_UNUSED(dev);
    GGML_LOG_ERROR("[ERROR] R-NPU device interface %s is not supported!\n", __func__);
    return nullptr;
}

static void ggml_backend_rnpu_device_event_free(
                ggml_backend_dev_t dev, ggml_backend_event_t event) {
    // TODO: support NPU Event in the future
    GGML_UNUSED(dev);
    GGML_UNUSED(event);
    GGML_LOG_ERROR("[ERROR] R-NPU device interface %s is not supported!\n", __func__);
}

static void ggml_backend_rnpu_device_event_synchronize(
                ggml_backend_dev_t dev, ggml_backend_event_t event) {
    // TODO: support NPU Event in the future
    GGML_UNUSED(dev);
    GGML_UNUSED(event);
    GGML_LOG_ERROR("[ERROR] R-NPU device interface %s is not supported!\n", __func__);
}

/**
 * @brief Structure defining the interface for R-NPU backend device
 *
 * The functions of the function pointers in the structure include
 * backend initialization, device information acquisition,
 * acquisition of the buffer type, device support test, and event management.
 */
const ggml_backend_device_i ggml_backend_rnpu_device_interface = {
    /* .get_name                = */ ggml_backend_rnpu_device_get_name,
    /* .get_description         = */ ggml_backend_rnpu_device_get_description,
    /* .get_memory              = */ ggml_backend_rnpu_device_get_memory,
    /* .get_type                = */ ggml_backend_rnpu_device_get_type,
    /* .get_props               = */ ggml_backend_rnpu_device_get_props,
    /* .init_backend            = */ ggml_backend_rnpu_device_init_backend,    // called for every card
    /* .get_buffer_type         = */ ggml_backend_rnpu_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_rnpu_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL, // not supported for rnpu
    /* .supports_op             = */ ggml_backend_rnpu_device_supports_op,
    /* .supports_buft           = */ ggml_backend_rnpu_device_supports_buft,
    /* .offload_op              = */ ggml_backend_rnpu_device_offload_op,
    /* .event_new               = */ ggml_backend_rnpu_device_event_new,
    /* .event_free              = */ ggml_backend_rnpu_device_event_free,
    /* .event_synchronize       = */ ggml_backend_rnpu_device_event_synchronize,
};