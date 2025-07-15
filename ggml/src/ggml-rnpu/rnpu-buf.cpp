#include "ggml-rnpu.h"
#include <cstdint>

#include "../ggml-backend-impl.h"

#include "./common.hpp"

// --------------------------------------------------------------------------------
// --------------------------- NPU buffer interface -------------------------------
// --------------------------------------------------------------------------------












const ggml_backend_buffer_i ggml_backend_rnpu_buffer_interface = {
    NULL
    // /* .free_buffer     = */ ggml_backend_rnpu_buffer_free_buffer,
    // /* .get_base        = */ ggml_backend_rnpu_buffer_get_base,
    // /* .init_tensor     = */ ggml_backend_rnpu_buffer_init_tensor,
    // /* .memset_tensor   = */ ggml_backend_rnpu_buffer_memset_tensor,
    // /* .set_tensor      = */ ggml_backend_rnpu_buffer_set_tensor,
    // /* .get_tensor      = */ ggml_backend_rnpu_buffer_get_tensor,
    // /* .cpy_tensor      = */ ggml_backend_rnpu_buffer_cpy_tensor,
    // /* .clear           = */ ggml_backend_rnpu_buffer_clear,
    // /* .reset           = */ NULL,
};