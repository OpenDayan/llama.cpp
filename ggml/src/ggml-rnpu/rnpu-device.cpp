#include "ggml-backend.h"
#include "ggml-rnpu.h"
// #include "ggml-backend.h"
#include "../ggml-backend-impl.h"

#include "./common.hpp"
#include "ggml.h"

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
    // TODO: confirm whether R-NPU support event properties.
    bool events = false;

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
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

static bool ggml_backend_rnpu_device_supports_op(
                ggml_backend_dev_t dev, const ggml_tensor * op) {
    // TODO: impl
    return false;
}

static bool ggml_backend_rnpu_device_supports_buft(
                ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // TODO: impl
    return false;
}

static bool ggml_backend_rnpu_device_offload_op(
                ggml_backend_dev_t dev, const ggml_tensor * op) {
    // TODO: impl
    return false;
}

static ggml_backend_event_t ggml_backend_rnpu_device_event_new(
                                ggml_backend_dev_t dev) {
    // TODO: impl
    return nullptr;
}

static void ggml_backend_rnpu_device_event_free(
                ggml_backend_dev_t dev, ggml_backend_event_t event) {
    // TODO: impl
}

static void ggml_backend_rnpu_device_event_synchronize(
                ggml_backend_dev_t dev, ggml_backend_event_t event) {
    // TODO: impl
}

/**
 * @brief Structure defining the interface for R-NPU backend device
 *
 * The functions of the function pointers in the structure include
 * backend initialization, device information acquisition,
 * acquisition of the buffer type, device support test, and event management.
 */
const ggml_backend_device_i ggml_backend_rnpu_device_interface = {
    // TODO:
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