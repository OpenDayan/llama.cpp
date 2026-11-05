#include "ggml-dayan-kernels.h"
#include "ggml.h"

#include <cstdio>
#include <initializer_list>

int main() {
    struct test_case {
        npu_runtime_topology_t topology;
        uint32_t cores;
        uint32_t clusters;
    };
    const test_case cases[] = {
        {{1, 1, 0x1, 0x1}, 1, 0},
        {{1, 4, 0x1, 0x1}, 1, 0},
        {{1, 4, 0x1, 0xf}, 4, 1},
        {{2, 4, 0x3, 0xf}, 8, 2},
        {{4, 4, 0xf, 0xf}, 16, 4},
        {{2, 4, 0x3, 0x7}, 6, 0},
        {{4, 4, 0xa, 0x5}, 4, 0},
        {{4, 4, 0xa, 0xf}, 8, 2},
    };
    for (const auto & test : cases) {
        DayanTopology topology;
        GGML_ASSERT(topology.init(test.topology));
        GGML_ASSERT(topology.cores == test.cores && topology.clusters == test.clusters);
        const auto single = topology.launch(false);
        const auto cooperative = topology.launch(true);
        GGML_ASSERT(single.task_mode == NPUTaskMode::kSingleCore);
        GGML_ASSERT(single.task_dim.x == test.cores && single.task_dim.y == 1 && single.task_dim.z == 1);
        GGML_ASSERT(cooperative.task_mode == (test.clusters ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore));
        GGML_ASSERT(cooperative.task_dim.x == (test.clusters ? 4 : test.cores));
        GGML_ASSERT(cooperative.task_dim.y == (test.clusters ? test.clusters : 1));
        GGML_ASSERT(topology.launch(false, 1).task_dim.x == 1);
    }
    for (const auto & info : {npu_runtime_topology_t{0, 4, 0, 1}, {1, 4, 2, 1}, {1, 4, 1, 16}, {1, 4, 1, 0}}) {
        DayanTopology topology;
        GGML_ASSERT(!topology.init(info));
    }
    for (uint32_t tokens : {2U, 5U, 1024U, 6145U}) {
        const auto single = DayanMoeDispatchLayout::make(tokens, 256, 8, 1);
        const auto four = DayanMoeDispatchLayout::make(tokens, 256, 8);
        GGML_ASSERT(single.position_stride >= (uint64_t) tokens * 8 * sizeof(uint32_t));
        GGML_ASSERT(single.offsets_offset >= single.positions_offset + single.position_stride);
        GGML_ASSERT(single.bytes <= four.bytes);
    }
    for (uint64_t width : {1, 16, 481, 513, 1541}) {
        for (uint32_t groups : {1, 2, 8}) {
            GGML_ASSERT(dayan_partition_begin(width, 0, groups) == 0);
            GGML_ASSERT(dayan_partition_begin(width, groups, groups) == width);
            for (uint32_t group = 0; group < groups; ++group) {
                const uint64_t begin = dayan_partition_begin(width, group, groups);
                const uint64_t end = dayan_partition_begin(width, group + 1, groups);
                GGML_ASSERT(begin <= end && end <= width);
                GGML_ASSERT(begin == width || begin % 64 == 0);
            }
        }
    }
    std::puts("test-dayan-topology: PASS");
}
