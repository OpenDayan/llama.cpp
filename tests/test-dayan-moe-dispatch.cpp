#include "ggml-dayan-kernels.h"
#include "ggml-dayan-common.h"
#include "ggml-dayan-moe.h"

#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
static void test_check(int status) { if (status < 0) { dayan_thread_exit(status); } }
#endif

__kernel__ void test_moe_dispatch(const DayanMxfp4MoeConfig * config, const int32_t * ids, void * workspace,
                                  void * input, void * grouped, void * output, uint32_t reverse_only) {
#if defined(__DAYAN_DEVICE__)
    using namespace ggml_dayan_matmul_detail;
    dayan_cache_invalidate(config, sizeof(*config));
    const auto & r = config->routing;
    const auto & mm = config->matmul;
    const auto layout = DayanMxfp4Layout::make(r.tokens * r.experts_used, mm.k);
    const auto input_layout = DayanMxfp4Layout::make(r.tokens * r.activation_slots, mm.k);
    if (!reverse_only) {
        DayanMoeSortConfig sort = {};
        sort.sort.rows = r.tokens;
        sort.sort.width = r.experts;
        sort.sort.output_stride = r.ids_row_stride / sizeof(int32_t);
        sort.experts_used = r.experts_used;
        sort.workspace_bytes = config->workspace_bytes;
        test_check(run_moe_dispatch(&sort, ids, workspace));
        test_check(dayan_cluster_sync());
        moe_permute_rows(config, workspace, input, (char *) input + input_layout.scale_offset,
                         grouped, (char *) grouped + layout.scale_offset, false);
        test_check(dayan_cluster_sync());
    }
    moe_permute_rows(config, workspace, output, (char *) output + layout.scale_offset,
                     grouped, (char *) grouped + layout.scale_offset, true);
    test_check(dayan_cluster_sync());
#endif
}

__kernel__ void test_moe_addresses(const uint32_t * rows, uint64_t * output) {
#if defined(__DAYAN_DEVICE__)
    using namespace ggml_dayan_matmul_detail;
    const uint64_t base = (uintptr_t) dayan_uram_base();
    memory_load_bytes(base, rows, 256);
    configure_vec_grid(2);
    configure_vec_iterator<kVecV0Base>(0, 0, 128, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 256, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    moe_address_table(base + 512, base, 0x81ffffff80ULL, 128);
    dayan_wait_vec();
    memory_store_bytes(output, base + 512, 512);
#endif
}

DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(test_moe_dispatch, 1, DAYAN_KERNEL_MAX_CHILD_THREADS,
    0, 128U * 1024U, 2, 0, 1, 0, 0, kDayanClusterResources);
DAYAN_KERNEL_TASK_DESC(test_moe_addresses, 1, 0, 0, 32768, 2, 0, 1, 0);

#if !defined(__DAYAN_DEVICE__)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static void check(int status, const char * operation) {
    if (status < 0) {
        std::fprintf(stderr, "%s: status=%d\n", operation, status);
        std::exit(1);
    }
}

static void wait(npu_event_t * event) {
    npu_event_info_t info;
    check(npu_event_wait(event, 300000, &info), "wait");
    check(info.status, "event");
    check(npu_event_destroy(event), "destroy event");
}

static npu_buffer_t * allocate(npu_context_t * context, size_t bytes) {
    npu_buffer_desc_t desc;
    npu_buffer_desc_init(&desc, bytes);
    desc.alignment = 256;
    npu_buffer_t * buffer = nullptr;
    check(npu_buffer_create(context, &desc, &buffer), "allocate");
    return buffer;
}

static void upload(npu_stream_t * stream, npu_buffer_t * buffer, const void * data, size_t bytes) {
    npu_event_t * event = nullptr;
    check(npu_memcpy_h2d_async(stream, buffer, 0, data, bytes, &event), "upload");
    wait(event);
}

static void download(npu_stream_t * stream, void * data, npu_buffer_t * buffer, size_t bytes) {
    npu_event_t * event = nullptr;
    check(npu_memcpy_d2h_async(stream, data, buffer, 0, bytes, &event), "download");
    wait(event);
}

static void equal(const void * actual, const void * expected, size_t bytes, const char * what) {
    if (std::memcmp(actual, expected, bytes)) {
        const auto * a = (const unsigned char *) actual;
        const auto * b = (const unsigned char *) expected;
        for (size_t i = 0; i < bytes; ++i) {
            if (a[i] != b[i]) {
                std::fprintf(stderr, "%s: byte=%zu actual=%u expected=%u\n", what, i, a[i], b[i]);
                break;
            }
        }
        std::exit(1);
    }
}

static void routing_case(npu_context_t * context, npu_stream_t * stream, uint32_t tokens, uint32_t experts,
                          uint32_t used, uint32_t slots, bool skew, uint32_t cores, uint32_t expert_shift = 0) {
    constexpr uint32_t columns = 160;
    const uint32_t total = tokens * used;
    const auto dispatch = DayanMoeDispatchLayout::make(tokens, experts, used, cores);
    const auto src_layout = DayanMxfp4Layout::make(tokens * slots, columns);
    const auto dst_layout = DayanMxfp4Layout::make(total, columns);
    DayanMxfp4MoeConfig config = {};
    config.routing = {columns, columns, experts, used, tokens, slots, (used + 3) * sizeof(int32_t)};
    config.matmul.k = config.matmul.n = columns;
    config.matmul.a_stride = src_layout.row_stride;
    config.matmul.a_scale_stride = src_layout.scale_stride;
    config.matmul.output_stride = dst_layout.row_stride;
    config.matmul.output_scale_stride = dst_layout.scale_stride;
    config.matmul.output_type = kDayanDataTypeMXFP4;
    config.workspace_bytes = dispatch.bytes;
    std::vector<int32_t> ids((size_t) tokens * (used + 3), -1);
    std::vector<uint32_t> counts(experts), offsets(experts + 1), mapping(total);
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t rank = 0; rank < used; ++rank) {
            const uint32_t expert = (expert_shift + (skew ? 0 : token * 17) + rank) % experts;
            ids[token * (used + 3) + rank] = expert;
            ++counts[expert];
        }
    }
    std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
    auto cursor = offsets;
    for (uint32_t core = 0; core < cores; ++core) {
        for (uint32_t token = core; token < tokens; token += cores) {
            for (uint32_t rank = 0; rank < used; ++rank) {
                mapping[token * used + rank] = cursor[ids[token * (used + 3) + rank]]++;
            }
        }
    }
    std::vector<unsigned char> input(src_layout.bytes), grouped(dst_layout.bytes), output(dst_layout.bytes);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = (i * 13 + i / 127) % 251; }
    npu_buffer_t * buffers[] = {allocate(context, sizeof(config)), allocate(context, ids.size() * 4),
        allocate(context, dispatch.bytes), allocate(context, input.size()), allocate(context, (size_t) total * columns * 4 + dst_layout.bytes),
        allocate(context, (size_t) total * columns * 4 + dst_layout.bytes)};
    upload(stream, buffers[0], &config, sizeof(config));
    upload(stream, buffers[1], ids.data(), ids.size() * 4);
    upload(stream, buffers[3], input.data(), input.size());
    npu_kernel_launch_attrs_t attrs;
    npu_kernel_launch_attrs_init(&attrs, context, stream, "moe-dispatch");
    attrs.config.task_dim = NPUTaskDim3{cores, 1, 1};
    attrs.config.task_mode = cores == 4 ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
    attrs.timeout_ms = 300000;
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        check(test_moe_dispatch(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], 0, &attrs), "dispatch");
        std::vector<unsigned char> workspace(dispatch.bytes);
        download(stream, workspace.data(), buffers[2], workspace.size());
        equal(workspace.data() + dispatch.offsets_offset, offsets.data(), offsets.size() * 4, "offsets");
        download(stream, grouped.data(), buffers[4], grouped.size());
        download(stream, output.data(), buffers[5], output.size());
        for (uint32_t token = 0; token < tokens; ++token) {
            for (uint32_t rank = 0; rank < used; ++rank) {
                const uint32_t row = token * used + rank, source = token * slots + rank % slots, group = mapping[row];
                equal(workspace.data() + dispatch.positions_offset + (token % cores) * dispatch.position_stride +
                      ((token / cores) * used + rank) * 4, &group, 4, "mapping");
                equal(grouped.data() + group * dst_layout.row_stride, input.data() + source * src_layout.row_stride, dst_layout.row_stride, "gather");
                equal(output.data() + row * dst_layout.row_stride, input.data() + source * src_layout.row_stride, dst_layout.row_stride, "scatter");
                for (unsigned plane = 0; plane < (columns + 127) / 128; ++plane) {
                    const auto * expected = input.data() + src_layout.scale_offset + plane * src_layout.scale_stride + source * 4;
                    equal(grouped.data() + dst_layout.scale_offset + plane * dst_layout.scale_stride + group * 4, expected, 4, "gather scales");
                    equal(output.data() + dst_layout.scale_offset + plane * dst_layout.scale_stride + row * 4, expected, 4, "scatter scales");
                }
            }
        }
    }
    config.matmul.output_type = kDayanDataTypeF32;
    config.matmul.output_stride = columns * 4;
    std::vector<uint32_t> fp32((size_t) total * columns), restored(fp32.size());
    std::iota(fp32.begin(), fp32.end(), 0U);
    upload(stream, buffers[0], &config, sizeof(config));
    upload(stream, buffers[4], fp32.data(), fp32.size() * 4);
    check(test_moe_dispatch(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], 1, &attrs), "scatter f32");
    download(stream, restored.data(), buffers[5], restored.size() * 4);
    for (uint32_t row = 0; row < total; ++row) {
        equal(restored.data() + row * columns, fp32.data() + mapping[row] * columns, columns * 4, "scatter f32");
    }
    for (auto * buffer : buffers) { check(npu_buffer_destroy(buffer), "free"); }
    std::printf("PASS tokens=%u experts=%u used=%u slots=%u skew=%d shift=%u (repeat, MXFP4 planes, F32)\n", tokens, experts, used, slots, skew, expert_shift);
    std::fflush(stdout);
}

int main() {
    npu_runtime_options_t options;
    npu_runtime_options_init(&options);
    if (const char * socket = std::getenv("DAYAN_SERVER_SOCKET")) { options.server_socket = socket; }
    npu_runtime_t * runtime = nullptr;
    npu_context_t * context = nullptr;
    npu_stream_t * stream = nullptr;
    check(npu_runtime_open(&runtime, &options), "open");
    check(npu_context_create(runtime, nullptr, &context), "context");
    check(npu_stream_create(context, nullptr, &stream), "stream");
    npu_runtime_topology_t info;
    check(npu_runtime_get_topology(runtime, &info), "topology");
    DayanTopology topology;
    if (!topology.init(info) || (topology.cores != 1 && topology.cores != 4)) { return 1; }
    const uint32_t cores = topology.cores;
    uint32_t rows[64];
    uint64_t addresses[64], expected[64];
    for (unsigned i = 0; i < 64; ++i) { rows[i] = i * 1023; expected[i] = 0x81ffffff80ULL + (uint64_t) rows[i] * 128; }
    auto * source = allocate(context, sizeof(rows));
    auto * destination = allocate(context, sizeof(addresses));
    upload(stream, source, rows, sizeof(rows));
    npu_kernel_launch_attrs_t attrs;
    npu_kernel_launch_attrs_init(&attrs, context, stream, "moe-addresses");
    attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
    attrs.config.task_mode = NPUTaskMode::kSingleCore;
    attrs.timeout_ms = 300000;
    check(test_moe_addresses(source, destination, &attrs), "addresses");
    download(stream, addresses, destination, sizeof(addresses));
    equal(addresses, expected, sizeof(expected), "64-bit addresses");
    check(npu_buffer_destroy(source), "free");
    check(npu_buffer_destroy(destination), "free");
    std::puts("PASS 64-bit address carry");
    routing_case(context, stream, 2, 65, 8, 1, false, cores);
    routing_case(context, stream, 5, 65, 8, 1, false, cores);
    routing_case(context, stream, 129, 256, 8, 8, true, cores);
    routing_case(context, stream, 129, 1025, 8, 1, false, cores);
    routing_case(context, stream, 5, 1089, 8, 1, true, cores, 1024);
    routing_case(context, stream, 6145, 4096, 8, 1, false, cores);
    routing_case(context, stream, 17, 63, 3, 1, false, cores);
    check(npu_stream_destroy(stream), "destroy stream");
    check(npu_context_destroy(context), "destroy context");
    check(npu_runtime_close(runtime), "close");
    return 0;
}
#endif
