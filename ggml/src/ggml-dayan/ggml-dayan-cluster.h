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
