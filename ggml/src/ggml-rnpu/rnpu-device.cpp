#include "ggml-rnpu.h"
// #include "ggml-backend.h"
#include "../ggml-backend-impl.h"

#include "./common.hpp"

// --------------------------------------------------------------------------------
// --------------------------- NPU device interface -------------------------------
// --------------------------------------------------------------------------------















/**
 * @brief Structure defining the interface for R-NPU backend device
 *
 * The functions of the function pointers in the structure include
 * backend initialization, device information acquisition,
 * acquisition of the buffer type, device support test, and event management.
 */
const ggml_backend_device_i ggml_backend_rnpu_device_interface = {
    NULL
    // TODO:
    // /* .get_name                = */ ggml_backend_rnpu_device_get_name,
    // /* .get_description         = */ ggml_backend_rnpu_device_get_description,
    // /* .get_memory              = */ ggml_backend_rnpu_device_get_memory,
    // /* .get_type                = */ ggml_backend_rnpu_device_get_type,
    // /* .get_props               = */ ggml_backend_rnpu_device_get_props,
    // /* .init_backend            = */ ggml_backend_rnpu_device_init,    // called for every card
    // /* .get_buffer_type         = */ ggml_backend_rnpu_device_get_buffer_type,
    // /* .get_host_buffer_type    = */ ggml_backend_rnpu_device_get_host_buffer_type,
    // /* .buffer_from_host_ptr    = */ NULL, // not supported for rnpu
    // /* .supports_op             = */ ggml_backend_rnpu_supports_op,
    // /* .supports_buft           = */ ggml_backend_rnpu_supports_buft,
    // /* .offload_op              = */ ggml_backend_rnpu_offload_op,
    // /* .event_new               = */ ggml_backend_rnpu_device_event_new,
    // /* .event_free              = */ ggml_backend_rnpu_device_event_free,
    // /* .event_synchronize       = */ ggml_backend_rnpu_device_event_synchronize,
};