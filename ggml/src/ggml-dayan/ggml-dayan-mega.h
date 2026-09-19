#pragma once

#include "ggml-dayan-graph.h"
#include "ggml-dayan-mega-executor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_backend_dayan_mega_runtime {
    void * context;
    bool   enabled;
    bool (*resolve)(void *, const void *, size_t, npu_buffer_t **, uint64_t *);
    int (*prepare_workspace)(void *, size_t, npu_buffer_t **);
    int (*launch)(void *, uint16_t, const void *, size_t, npu_buffer_t * const *);
};

struct ggml_backend_dayan_mega_result {
    int    status;
    size_t end;
    bool   launched;
};

ggml_backend_dayan_mega_result ggml_backend_dayan_try_qwen35_mega(
    const ggml_backend_dayan_mega_runtime &            runtime,
    const std::vector<ggml_backend_dayan_graph_slot> & slots,
    size_t                                              begin);
