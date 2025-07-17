#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rnpu.h"

#include "./runtime-host.hpp"

#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"
#include "./common.hpp"
#include "ggml.h"

// ------------------------------------------------------------------------------------
// ------------------------ NPU host buffer type internal API -------------------------
// ------------------------------------------------------------------------------------

/**
 * @brief Retrieve the descriptive name of a R-NPU host buffer type
 *
 * @param buft Pointer to the host buffer type.
 * @return Pointer to the C-style string containing the name.
 */
static const char * ggml_backend_rnpu_host_buffer_type_name(
                                ggml_backend_buffer_type_t buft) {
    return GGML_RNPU_NAME "_Host";

    GGML_UNUSED(buft);
}

/**
 * @brief Free resources of a R-NPU host buffer
 *
 * @param buffer Pointer to the host buffer
 */
static void ggml_backend_rnpu_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    // TODO: buffer->context->host_prt ???
    NPU_CHECK(npuFreeHost(buffer->context));
}

/**
 * @brief Allocate a new R-NPU host buffer of the given size.
 *
 * @param size Size in bytes of the host buffer to be allocated.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
static void * ggml_rnpu_host_malloc(size_t size) {
    if (getenv("GGML_RNPU_NO_PINNED") != nullptr) {
        return nullptr;
    }

    // TODO: confirm NPU host memory alignment
    const size_t alignment = 128;
    size = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }

    void * hostPtr = nullptr;
    npuErr_t err = npuMallocHost((void **) &hostPtr, size);
    if (err != npuSuccess) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, npuGetErrorString(err));
        return nullptr;
    }
    return hostPtr;
}

/**
 * @brief Allocates a new R-NPU host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to allocated host buffer, or CPU buffer if allocation fails
 */
static ggml_backend_buffer_t ggml_backend_rnpu_host_buffer_type_alloc_buffer(
                                    ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_rnpu_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(buft, size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_rnpu_host_buffer_free_buffer;

    return buffer;
}



// ----------------------------------------------------------------------------------------
// -------------------------- NPU host buffer type external API ---------------------------
// ----------------------------------------------------------------------------------------

/**
 * @brief Interface for managing R-NPU host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for R-NPU buffer types. The structure of buffer type is initialized
 * once and reused on subsequent calls.
 *
 * @return Pointer to initialized host buffer type.
 */
ggml_backend_buffer_type_t ggml_backend_rnpu_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_rnpu_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_rnpu_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_rnpu_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_rnpu_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_rnpu_buffer_type_host;
}