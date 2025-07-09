#include "ggml-alloc.h"
#include "ggml-rnpu.h"

#include <cstddef>
#include <string>

#include "ggml-backend.h"
#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"

#include "./common.hpp"

// ------------------------------------------------------------------------------
// ---------------------------- NPU backend interface ---------------------------
// ------------------------------------------------------------------------------











/**
 * @brief Structure defining the interface for R-NPU backend.
 *
 * This structure contains function pointers for various operations
 * supported by the R-NPU backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static const ggml_backend_i ggml_backend_rnpu_interface = {
    // TODO:
    // /* .get_name                = */ ggml_backend_rnpu_name,
    // /* .free                    = */ ggml_backend_rnpu_free,
    // /* .set_tensor_async        = */ ggml_backend_rnpu_set_tensor_async,
    // /* .get_tensor_async        = */ ggml_backend_rnpu_get_tensor_async,
    // /* .cpy_tensor_async        = */ ggml_backend_rnpu_cpy_tensor_async,
    // /* .synchronize             = */ ggml_backend_rnpu_synchronize,
    // /* .graph_plan_create       = */ NULL,
    // /* .graph_plan_free         = */ NULL,
    // /* .graph_plan_update       = */ NULL,
    // /* .graph_plan_compute      = */ NULL,
    // /* .graph_compute           = */ ggml_backend_rnpu_graph_compute,
    // /* .event_record            = */ ggml_backend_rnpu_event_record,
    // /* .event_wait              = */ ggml_backend_rnpu_event_wait,
};


// --------------------------------------------------------------------------------
// --------------------------- NPU backend internal API ---------------------------
// --------------------------------------------------------------------------------

/**
 * @brief Return GUID of R-NPU.
 *
 * @return Pointer to GUID of R-NPU.
 */
static ggml_guid_t ggml_backend_rnpu_guid() {
    // TODO: confirm NPU guid
    static ggml_guid guid = { 0x0 };
    return &guid;
}


// --------------------------------------------------------------------------------
// --------------------------- NPU backend external API ---------------------------
// --------------------------------------------------------------------------------

/**
 * @brief Check whether backend is R-NPU backend.
 *
 * @param backend Pointer to given backend.
 * @return 'True' if backend's GUID is R-NPU GUID.
 */
bool ggml_backend_is_rnpu(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_rnpu_guid());
}

/**
 * @brief Retrieve device count from R-NPU device information
 * @return Number of available NPUs
 */
int32_t  ggml_backend_rnpu_get_device_count() {
    return ggml_rnpu_info().device_count;
}

/**
 * @brief Retrieve description of a given device.
 *
 * @param device The device ID.
 * @param description The description string to be set.
 * @param description_size The description string buffer size.
 */
void ggml_backend_rnpu_get_device_description(
    int device, char * description, size_t description_size) {
    ggml_rnpu_set_device(device);
    // TODO: NPU property API
    // npuDeivceProp prop;
    // NPU_CHECK(npuGetDeviceProperties(&prop, i))
    // snprintf(description, description_size, "%s", prop.name);
}

/**
 * @brief Retrieve memory size of a given device.
 *
 * @param device The device ID.
 * @param free The free memory size on the device to be set.
 * @param total The total memory size on the device to be set.
 */
void ggml_backend_rnpu_get_device_memory(
    int device, size_t * free, size_t * total) {
    ggml_rnpu_set_device(device);

    // TODO: mem info API
    // NPU_CHECK(npuMemGetInfo(free, total))
}

/**
 * @brief Initialize a backend for a NPU device.
 *
 * @param device The device ID.
 * @return Pointer to initialized backend.
 */
ggml_backend_t ggml_backend_rnpu_init(int32_t device) {
    if (device < 0 || device >= ggml_backend_rnpu_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_rnpu_context * ctx  = new ggml_backend_rnpu_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t rnpu_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_rnpu_guid(),
        /* .interface = */ ggml_backend_rnpu_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_rnpu_reg(), device),
        /* .context   = */ ctx,
    };

    return rnpu_backend;
}