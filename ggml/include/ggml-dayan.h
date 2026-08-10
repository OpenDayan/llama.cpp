#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_DAYAN_NAME "DAYAN"

GGML_BACKEND_API ggml_backend_t     ggml_backend_dayan_init(void);
GGML_BACKEND_API bool               ggml_backend_is_dayan(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_dayan_reg(void);
GGML_BACKEND_API bool               ggml_backend_dayan_graph_rewrite(ggml_backend_t        backend,
                                                                     struct ggml_context * ctx,
                                                                     struct ggml_cgraph *  cgraph);

#ifdef __cplusplus
}
#endif
