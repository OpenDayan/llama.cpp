#include "ggml-alloc.h"
#include "ggml-rnpu.h"
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "./runtime-wrapper.hpp"

#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"

#include "./common.hpp"
#include "ggml.h"


// --------------------------------------------------------------------------------------
// ----------------------------- NPU buffer type interface ------------------------------
// --------------------------------------------------------------------------------------

/**
 * @brief Retrieves the name associated with a R-NPU buffer type.
 *
 * @param buft Pointer to the buffer type context.
 * @return Pointer to the C-style string containing the name.
 */
static const char* ggml_backend_rnpu_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    ggml_backend_rnpu_buffer_type_context* buft_ctx =
        (ggml_backend_rnpu_buffer_type_context*)buft->context;

    return buft_ctx->name.c_str();
}

/**
 * @brief Allocates a new R-NPU buffer of the specified type and size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
static ggml_backend_buffer_t
ggml_backend_rnpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_rnpu_buffer_type_context* buft_ctx =
        (ggml_backend_rnpu_buffer_type_context*)buft->context;

    ggml_rnpu_set_device(buft_ctx->device);

    // TODO: Confirm if alloc alignment is needed.
    const size_t alignment = 128;
    size = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }

    void * dev_ptr = NULL;
    npuErr_t err = npuMalloc(&dev_ptr, size);
    if (err != npuSuccess) {
        // clear the error
        (void)npuGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: npuMalloc failed: %s\n",
                __func__, size / 1024.0 / 1024.0, buft_ctx->device, npuGetErrorString(err));
        return nullptr;
    }

    ggml_backend_rnpu_buffer_context * ctx =
        new ggml_backend_rnpu_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_rnpu_buffer_interface, ctx, size);
}

/**
 * @brief Retrieves the memory alignment for R-NPU buffers of this type.
 *
 * This function returns the alignment requirement in bytes for memory allocated
 * by the R-NPU buffer type.
 *
 * @param buft Pointer to the buffer type context. (Unused)
 * @return The alignmnet requirement in bytes. (fixed at 128 in R-NPU)
 */
static size_t ggml_backend_rnpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // TODO: Confirm alignment requirement in bytes.
    GGML_UNUSED(buft);
    return 128;
}

/**
 * @brief Calculates the allocation size required for a tensor in a R-NPU buffer.
 *
 * Computes the total allocation size needed for storing the tensor's data in a
 * R-NPU buffer, considering any necessary padding or adjustments for quantized
 * types.
 *
 * @param buft Pointer the buffer type context. (Unused)
 * @param tensor Pointer to the given tensor.
 * @return The total allocation size in bytes required for the tensor in the
 * R-NPU buffer.
 */
static size_t ggml_backend_rnpu_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    GGML_UNUSED(buft);

    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // TODO: Confirm MATRIX_ROW_PADDING is 512
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_rnpu_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rnpu_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rnpu_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rnpu_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_rnpu_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
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

/**
 * @brief Check whether backend buffer type is NPU buffer type.
 *
 * @param buft Pointer to backend buffer type to be checked.
 * @return Return 'true' if buffer type is NPU buffer type.
 */
bool ggml_backend_buft_is_rnpu(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_rnpu_buffer_type_name;
}