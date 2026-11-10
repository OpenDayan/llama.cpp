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

#include <stdint.h>

static constexpr uint32_t kDayanClusterResources = 24;

struct DayanMatmulPhases {
    uint64_t mat = 0;
    uint64_t post = 0;
};

struct DayanClusterState {
    struct alignas(64) Core {
        uint64_t uram;
        DayanMatmulPhases phases;
        uint64_t shared_phases;
        uint64_t shared_mask;
        uint32_t owner;
    } core[4];
};

static_assert(sizeof(DayanClusterState) == 256, "Cluster metadata must use one cache line per core");

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>

static inline DayanClusterState & dayan_cluster_state(uint64_t workspace) {
    return *(DayanClusterState *) (workspace ? (void *) (uintptr_t) workspace : shared);
}

static inline void dayan_cluster_flush(uint64_t workspace, uint32_t core) {
    if (workspace) {
        const auto & entry = dayan_cluster_state(workspace).core[core];
        dayan_cache_flush(&entry, sizeof(entry));
    }
}

static inline void dayan_cluster_invalidate(uint64_t workspace, uint32_t core) {
    if (workspace) {
        const auto & entry = dayan_cluster_state(workspace).core[core];
        dayan_cache_invalidate(&entry, sizeof(entry));
    }
}

static inline int dayan_cluster_sync() {
    if (dayan_task_mode() == 1 && dayan_task_dim(DAYAN_TASK_X) == 1) { return 0; }
    const auto mine = DAYAN_SHARED_RESOURCE(20 + dayan_task_id(DAYAN_TASK_X));
    const uint64_t all = 15ULL << DAYAN_SHARED_RESOURCE(20);
    // Two generations return the tokens to phase zero. Each core consumes every arrival.
    for (int phase = 0; phase < 2; ++phase) {
        int status = dayan_resource_wait_free(mine);
        if (status < 0) { return status; }
        status = dayan_resource_produce_lsu(mine, 4, 0);
        if (status < 0) { return status; }
        status = dayan_resource_wait(0, all, phase ? all : 0, 0);
        if (status < 0) { return status; }
        status = dayan_resource_consume_lsu(all);
        if (status < 0) { return status; }
        dayan_wait_lsu();
    }
    return 0;
}
#endif
