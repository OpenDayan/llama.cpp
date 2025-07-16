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
 * @brief Retrieves the name associated with the R-NPU backend.
 *
 * This function returns the name assigned to the R-NPU backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the R-NPU backend structure.
 * @return Pointer to a constant C-style string representing the backend name.
 */
static const char * ggml_backend_rnpu_get_name(ggml_backend_t backend) {
    ggml_backend_rnpu_context * rnpu_ctx = (ggml_backend_rnpu_context *)backend->context;

    return rnpu_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the R-NPU backend.
 *
 * This function releases resources associated with the R-NPU backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the R-NPU backend structure to be freed.
 */
static void ggml_backend_rnpu_free(ggml_backend_t backend) {
    ggml_backend_rnpu_context * rnpu_ctx = (ggml_backend_rnpu_context *)backend->context;

    delete rnpu_ctx;
    delete backend;
}

/**
 * @brief Sets tensor data asynchronously in the R-NPU backend.
 *
 * @param backend Pointer to the R-NPU backend structure.
 * @param tensor Pointer to the tensor structure to set data for.
 * @param data Pointer to the host data to copy to the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_rnpu_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rnpu_context * rnpu_ctx = (ggml_backend_rnpu_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_rnpu_buffer_type(rnpu_ctx->device) && "unsupported buffer type");

    // TODO: MemcpyAsync API
    // NPU_CHECK(rnpuMemcpyAsync((char *)tensor->data + offset, data, size, rnpuMemcpyHostToDevice, rnpu_ctx->stream()));
    GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);
}

/**
 * @brief Gets tensor data asynchronously in the R-NPU backend.
 *
 * @param backend Pointer to the R-NPU backend structure.
 * @param tensor Pointer to the tensor structure to get data from.
 * @param data Pointer to the host data to copy from the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_rnpu_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rnpu_context * rnpu_ctx = (ggml_backend_rnpu_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_rnpu_buffer_type(rnpu_ctx->device) && "unsupported buffer type");

    // TODO: MemcpyAsync API
    // NPU_CHECK(rnpuMemcpyAsync(data, (const char *)tensor->data + offset, size, rnpuMemcpyDeviceToHost, rnpu_ctx->stream()));
    GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);
}

/**
 * @brief Asynchronously copies tensor data between R-NPU backends.
 *
 * This function copies tensor data asynchronously between two R-NPU backends. It
 * checks if both tensors reside in R-NPU buffers and whether the devices support
 * peer-to-peer access for direct copying. If not, it returns false.
 *
 * @param backend_src Pointer to the source R-NPU backend structure.
 * @param backend_dst Pointer to the destination R-NPU backend structure.
 * @param src Pointer to the source tensor to copy data from.
 * @param dst Pointer to the destination tensor to copy data to.
 * @return true if the copy operation succeeds, false otherwise.
 */
static bool ggml_backend_rnpu_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_rnpu(backend_src) || !ggml_backend_is_rnpu(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_rnpu(src->buffer) || !ggml_backend_buffer_is_rnpu(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_rnpu_context * rnpu_ctx_src = (ggml_backend_rnpu_context *)backend_src->context;
    ggml_backend_rnpu_context * rnpu_ctx_dst = (ggml_backend_rnpu_context *)backend_dst->context;

    ggml_backend_rnpu_buffer_context * buf_ctx_src = (ggml_backend_rnpu_buffer_context *)buf_src->context;
    ggml_backend_rnpu_buffer_context * buf_ctx_dst = (ggml_backend_rnpu_buffer_context *)buf_dst->context;

    GGML_ASSERT(rnpu_ctx_src->device == buf_ctx_src->device);
    GGML_ASSERT(rnpu_ctx_dst->device == buf_ctx_dst->device);

    if (backend_src != backend_dst) {
        // copy on src stream
        if (rnpu_ctx_src->device == rnpu_ctx_dst->device) {
            // TODO: MemcpyAsync API
            // NPU_CHECK(rnpuMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), rnpuMemcpyDeviceToDevice, rnpu_ctx_src->stream()));
            GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);
        } else {
            // TODO: MemcpyPeerAsync API
            // NPU_CHECK(rnpuMemcpyPeerAsync(dst->data, rnpu_ctx_dst->device, src->data, rnpu_ctx_src->device,
            //                                ggml_nbytes(dst), rnpu_ctx_src->stream()));
            GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);
        }
    } else {
        // src and dst are on the same backend
        // TODO: MemcpyAsync API
        // NPU_CHECK(rnpuMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), rnpuMemcpyDeviceToDevice, rnpu_ctx_src->stream()));
        GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);
    }
    return true;
}

/**
 * @brief Synchronizes a R-NPU backend.
 *
 * This function synchronizes the specified R-NPU backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the R-NPU backend structure to synchronize.
 */
static void ggml_backend_rnpu_synchronize(ggml_backend_t backend) {
    ggml_backend_rnpu_context * rnpu_ctx = (ggml_backend_rnpu_context *)backend->context;

    // TODO: Stream support
    // NPU_CHECK(rnpuStreamSynchronize(rnpu_ctx->stream()));
    GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);

    GGML_UNUSED(backend);
}

/**
 * @brief Computes a computational graph using a R-NPU backend.
 *
 * This function computes the operations defined in the computational graph
 * using the specified R-NPU backend.
 *
 * @param backend Pointer to the R-NPU backend structure to use for computation.
 * @param cgraph Pointer to the computational graph structure containing nodes
 *               representing operations to be computed.
 * @return enum ggml_status Returns GGML_STATUS_SUCCESS if computation
 *         completes successfully, otherwise an appropriate error status.
 */
 static enum ggml_status ggml_backend_rnpu_graph_compute(
                            ggml_backend_t backend, ggml_cgraph* cgraph) {
    ggml_backend_rnpu_context* rnpu_ctx =
        (ggml_backend_rnpu_context*)backend->context;

    ggml_rnpu_set_device(rnpu_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }

        bool ok = ggml_rnpu_compute_forward(*rnpu_ctx, node);

        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__,
                    node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Records an event on the R-NPU backend stream.
 *
 * This function records the given event on the NPU runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
 static void ggml_backend_rnpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rnpu_context* rnpu_ctx =
        (ggml_backend_rnpu_context*)backend->context;
    // TODO: Event API
    // NPU_CHECK(npuRecordEvent((npuEvent)event->context, rnpu_ctx->stream()));
    GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);
}

/**
 * @brief Waits for a recorded event to complete on the rnpu backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * NPU runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
 static void ggml_backend_rnpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rnpu_context* rnpu_ctx =
        (ggml_backend_rnpu_context*)backend->context;
    if (ggml_backend_is_rnpu(backend)) {
        // TODO: Event API
        // NPU_CHECK(npuStreamWaitEvent(rnpu_ctx->stream(), (npuEvent)event->context));
        GGML_LOG_ERROR("[ERROR] R-NPU backend interface %s is not supported!\n", __func__);
    } else {
        GGML_ABORT("fatal error");
    }
}


/**
 * @brief Structure defining the interface for R-NPU backend.
 *
 * This structure contains function pointers for various operations
 * supported by the R-NPU backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static const ggml_backend_i ggml_backend_rnpu_interface = {
    /* .get_name                = */ ggml_backend_rnpu_get_name,
    /* .free                    = */ ggml_backend_rnpu_free,
    /* .set_tensor_async        = */ ggml_backend_rnpu_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rnpu_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_rnpu_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_rnpu_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rnpu_graph_compute,
    /* .event_record            = */ ggml_backend_rnpu_event_record,
    /* .event_wait              = */ ggml_backend_rnpu_event_wait,
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