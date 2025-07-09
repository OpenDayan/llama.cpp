#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rnpu_reg(void);

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rnpu_init(int32_t device);

GGML_BACKEND_API bool ggml_backend_is_rnpu(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rnpu_buffer_type(int32_t device);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rnpu_host_buffer_type(void);


GGML_BACKEND_API int32_t  ggml_backend_rnpu_get_device_count(void);
GGML_BACKEND_API void ggml_backend_rnpu_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_rnpu_get_device_memory(int device, size_t * free, size_t * total);

#ifdef  __cplusplus
}
#endif
