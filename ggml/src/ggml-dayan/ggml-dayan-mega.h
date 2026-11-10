/*
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
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
