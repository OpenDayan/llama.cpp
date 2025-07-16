#include "ggml-rnpu.h"
#include <cstdint>

#include "../ggml-backend-impl.h"

#include "./common.hpp"

// --------------------------------------------------------------------------------
// --------------------------- NPU buffer interface -------------------------------
// --------------------------------------------------------------------------------

/**
 * @brief Free resources associated with a R-NPU buffer (including its context).
 *
 * @param buffer Pointer to the R-NPU buffer to free.
 */
static void ggml_backend_rnpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rnpu_buffer_context * ctx =
        (ggml_backend_rnpu_buffer_context *)buffer->context;
    delete ctx;
}

/**
 * @brief Retrieve the base pointer of a R-NPU buffer.
 *
 * This function returns the base pointer of the R-NPU buffer, which points to
 * the device memory allocated for the buffer.
 *
 * @param buffer Pointer to the R-NPU buffer whose base pointer is to be retrieved.
 * @return Pointer to the base of the device memory allocated for the buffer.
 */
static void* ggml_backend_rnpu_buffer_get_base(
    ggml_backend_buffer_t buffer) {
    ggml_backend_rnpu_buffer_context * ctx =
        (ggml_backend_rnpu_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

/**
 * @brief Initialize a tensor using data from a R-NPU buffer.
 *
 * This function initializes a tensor using data from a R-NPU buffer. This function
 * handles special case for quantization and views.
 *
 * @param buffer Pointer to the R-NPU buffer to initialize the tensor with.
 * @param tensor Pointer to the tensor to initialize.
 * @return Initialization state. GGML_STATUS_SUCCESS if initialization was successful.
 */
static enum ggml_status ggml_backend_rnpu_buffer_init_tensor(
                            ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rnpu_buffer_context * ctx =
        (ggml_backend_rnpu_buffer_context *)buffer->context;

    if (tensor->view_src) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr) {
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = buffer->buft->iface.get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_rnpu_set_device(ctx->device);
            // TODO: Memset API
            //NPU_CHECK(npuMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Set tensor data to a specified value.
 *
 * @param buffer Pointer to the R-NPU buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param value The byte value to be set.
 * @param offset Offset in the tensor data in bytes from where to start setting.
 * @param size  Size of the data to be set in bytes.
 */
static void ggml_backend_rnpu_buffer_memset_tensor(
                ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                uint8_t value, size_t offset, size_t size) {
    ggml_backend_rnpu_buffer_context * ctx =
        (ggml_backend_rnpu_buffer_context *)buffer->context;

    ggml_rnpu_set_device(ctx->device);
    // TODO: Use npuMemsetAsync with npuStream to optimize
    // TODO: Memset API
    // NPU_CHECK(npuMemset((char *)tensor->data + offset, value, size));
}

/**
 * @brief Set tensor data from the given source data.
 *
 * @param buffer Pointer to the R-NPU buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param data Pointer to the source data to be copied into the tensor.
 * @param offset Offset in the source data from where to start copying.
 * @param size Size of the data to be copied in bytes.
 */
static void ggml_backend_rnpu_buffer_set_tensor(
                ggml_backend_buffer_t buffer, ggml_tensor *tensor,
                const void *data, size_t offset, size_t size) {
    ggml_backend_rnpu_buffer_context * ctx =
        (ggml_backend_rnpu_buffer_context *)buffer->context;

    ggml_rnpu_set_device(ctx->device);
    // TODO: Use npuMemcpyAsync with npuStream to optimize
    // TODO: Memcpy API
    // NPU_CHECK(npuMemcpy((char *)tensor->data + offset, data, size, npuMemcpyHostToDevice));
}

/**
 * @brief Get tensor data from the given source data.
 *
 * @param buffer Pointer to the R-NPU buffer where to retrieve the tensor data.
 * @param tensor Pointer to the tensor whose data will be retrieved.
 * @param data Pointer to the destination data to be copied into.
 * @param offset Offset in the destination data from where to start copying.
 * @param size Size of the data to be copied in bytes.
 */
static void ggml_backend_rnpu_buffer_get_tensor(
                ggml_backend_buffer_t buffer, const ggml_tensor *tensor,
                void *data, size_t offset, size_t size) {
    ggml_backend_rnpu_buffer_context * ctx =
        (ggml_backend_rnpu_buffer_context *)buffer->context;

    ggml_rnpu_set_device(ctx->device);
    // TODO: Use npuMemcpyAsync with npuStream to optimize
    // TODO: Memcpy API
    // NPU_CHECK(npuMemcpy(data, (char *)tensor->data + offset, size, npuMemcpyDeviceToHost));
}

/**
 * @brief Copy tensor data between R-NPU buffers if possible.
 *
 * @param buffer The destination R-NPU buffer the tensor data will be copied into.
 * @param src Pointer to the source tensor whose data will be copied.
 * @param dst Pointer to the destination tensor where the data will be copied into.
 * @return Return true if the copy operation succeeded, false otherwise.
 */
static bool ggml_backend_rnpu_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    if (ggml_backend_buft_is_rnpu(buffer->buft)) {
        ggml_backend_rnpu_buffer_context * src_ctx =
            (ggml_backend_rnpu_buffer_context *)src->buffer->context;

        ggml_backend_rnpu_buffer_context * dst_ctx =
            (ggml_backend_rnpu_buffer_context *)dst->buffer->context;

        size_t memcpy_size = ggml_nbytes(src);

        if (src_ctx->device == dst_ctx->device) {
            // TODO: Use npuMemcpyAsync with npuStream to optimize
            // TODO: Memcpy API
            // NPU_CHECK(npuMemcpy((char *)dst->data, memcpy_size,
            //                     (const char *)src->data, memcpy_size, npuMemcpyDeviceToDevice))
        } else {
            // TODO: support different device access by peer
            // TODO: peer API
            // int32_t canAccessPeer = 0;
            // NPU_CHECK(npuDeviceCanAccessPeer(&canAccessPeer, src_ctx->device,
            //                                  dst_ctx->device));
            // if (canAccessPeer) {
            //     ggml_rnpu_set_device(src_ctx->device);
            //     NPU_CHECK(npuDeviceEnablePeerAccess(dst_ctx->device, 0));
            //     NPU_CHECK(npuMemcpy((char *)dst->data, memcpy_size,
            //                         (const char *)src->data, memcpy_size,
            //                         npuMemcpyDeviceToDevice))
            //     return true;
            // }
        }
    }
    return false;
}

/**
 * @brief Clear a R-NPU buffer by setting all its memory to a specified value.
 *
 * @param buffer The R-NPU buffer to be cleared.
 * @param value The value to which each byte in the buffer will be set.
 */
static void ggml_backend_rnpu_buffer_clear(
    ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rnpu_buffer_context* ctx =
        (ggml_backend_rnpu_buffer_context*)buffer->context;

    ggml_rnpu_set_device(ctx->device);
    // TODO: Memset API
    // NPU_CHECK(npuDeviceSynchronize())
    // NPU_CHECK(npuMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
    // NPU_CHECK(npuDeviceSynchronize())
}

const ggml_backend_buffer_i ggml_backend_rnpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rnpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rnpu_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rnpu_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rnpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rnpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rnpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rnpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rnpu_buffer_clear,
    /* .reset           = */ NULL,
};

// -----------------------------------------------------------------------------------
// --------------------------- NPU buffer external API -------------------------------
// -----------------------------------------------------------------------------------

/**
 * @brief Check whether backend buffer is a R-NPU buffer.
 *
 * @param buft Pointer to backend buffer to be checked.
 * @return Return 'true' if buffer is a R-NPU buffer type.
 */
bool ggml_backend_buffer_is_rnpu(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_rnpu(buffer->buft);
}
