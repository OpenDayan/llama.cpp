#include "ggml-dayan-kernels.h"
#include "ggml-dayan-common.h"
#include "ggml-dayan-memory.h"

#include <dayan_kernel.h>

enum : uint32_t { kStride = 512, kRows = 3, kBytes = kRows * kStride, kOffset = 64 };

__kernel__ void test_matrix_dma(const void * input, void * output, void * snapshot,
                               uint32_t major_left, uint32_t major_tile,
                               uint32_t minor_left, uint32_t minor_tile) {
#if defined(__DAYAN_DEVICE__)
    using namespace ggml_dayan_matmul_detail;
    const uint64_t uram = (uintptr_t) dayan_uram_base();
    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, kBytes, kBytes, kBytes);
    load_matrix<0>(kBytes, 1, kBytes, 1, uram, (uintptr_t) output);
    dayan_wait_lsu();
    set_lsu_matrix<0>(checksum, kRows, kStride, kStride, kStride);
    load_matrix<0>(major_left, minor_left, major_tile, minor_tile, uram + kOffset, (uintptr_t) input + kOffset);
    dayan_wait_lsu();
    store_matrix<0>(major_left, minor_left, major_tile, minor_tile, uram + kOffset, (uintptr_t) output + kOffset);
    dayan_wait_lsu();
    set_lsu_matrix<0>(checksum, 1, kBytes, kBytes, kBytes);
    store_matrix<0>(kBytes, 1, kBytes, 1, uram, (uintptr_t) snapshot);
    dayan_wait_lsu();
#endif
}

__kernel__ void test_scalar_div(const float * input, const float * divisors, float * output,
                               uint32_t width, uint32_t rows, uint32_t right_rows, uint32_t right_stride) {
#if defined(__DAYAN_DEVICE__)
    DayanBinaryConfig config = {};
    config.elements = (uint64_t) width * rows;
    config.ne[0] = width;
    config.ne[1] = rows;
    config.ne[2] = config.ne[3] = 1;
    config.right_ne[0] = config.right_ne[2] = config.right_ne[3] = 1;
    config.right_ne[1] = right_rows;
    if (right_stride != sizeof(float)) {
        config.right_nb[0] = right_stride;
        config.right_nb[1] = config.right_nb[2] = (uint64_t) right_stride * right_rows;
    }
    ggml_dayan_matmul_detail::run_div_f32(&config, input, divisors, output);
#endif
}

DAYAN_KERNEL_TASK_DESC(test_matrix_dma, 1, 0, 0, 32768, 1, 0, 0, 0);
DAYAN_KERNEL_TASK_DESC(test_scalar_div, 1, 2, 0, DAYAN_NATIVE_TASK_URAM_BYTES, 2, 0, 1, 0);

#if !defined(__DAYAN_DEVICE__)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    npu_kernel_launch_attrs_t attrs;
    npu_kernel_launch_attrs_init(&attrs, context, stream, "dma-boundaries");
    attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
    attrs.config.task_mode = NPUTaskMode::kSingleCore;
    attrs.timeout_ms = 300000;

    std::vector<unsigned char> input(kBytes), guard(kBytes, 0xa5), output(kBytes), snapshot(kBytes);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = (i * 13 + i / kStride) % 251; }
    npu_buffer_t * buffers[] = {allocate(context, kBytes), allocate(context, kBytes), allocate(context, kBytes)};
    upload(stream, buffers[0], input.data(), kBytes);
    struct DmaCase { uint32_t left, tile, rows, tile_rows; };
    const DmaCase cases[] = {
        {4, 4, 1, 1}, {1, 1, 3, 3}, {32, 32, 3, 3}, {63, 63, 3, 3},
        {64, 64, 3, 3}, {65, 65, 3, 3}, {68, 68, 3, 3}, {127, 127, 3, 3},
        {128, 128, 3, 3}, {129, 129, 3, 3}, {257, 4, 3, 3}, {257, 65, 3, 3},
        {257, 128, 3, 3}, {4, 256, 3, 3}, {65, 256, 2, 3}, {129, 256, 3, 2},
        {0, 64, 3, 3}, {64, 0, 3, 3}, {64, 64, 0, 3}, {64, 64, 3, 0},
    };
    for (const auto & c : cases) {
        upload(stream, buffers[1], guard.data(), kBytes);
        check(test_matrix_dma(buffers[0], buffers[1], buffers[2], c.left, c.tile, c.rows, c.tile_rows, &attrs), "matrix DMA");
        download(stream, output.data(), buffers[1], kBytes);
        download(stream, snapshot.data(), buffers[2], kBytes);
        const uint32_t bytes = std::min(c.left, c.tile), rows = std::min(c.rows, c.tile_rows);
        for (uint32_t i = 0; i < kBytes; ++i) {
            const uint32_t row = i / kStride, column = i % kStride;
            const bool valid = row < rows && column >= kOffset && column < kOffset + bytes;
            const bool padding = row < rows && column >= kOffset + bytes && column < kOffset + ((bytes + 63) & ~63U);
            const unsigned char expected = valid ? input[i] : guard[i];
            const unsigned char loaded = padding ? 0 : expected;
            if (output[i] != expected || snapshot[i] != loaded) {
                std::fprintf(stderr, "DMA left=%u tile=%u rows=%u/%u byte=%u output=%u/%u URAM=%u/%u\n",
                             c.left, c.tile, c.rows, c.tile_rows, i, output[i], expected, snapshot[i], loaded);
                return 1;
            }
        }
        std::printf("PASS DMA left=%u tile=%u rows=%u/%u (data, padding, guards)\n", c.left, c.tile, c.rows, c.tile_rows);
        std::fflush(stdout);
    }
    for (auto * buffer : buffers) { check(npu_buffer_destroy(buffer), "free"); }

    struct DivCase { uint32_t width, rows, right_rows, stride, tasks; };
    const DivCase div_cases[] = {
        {8, 1, 1, 1, 1}, {8, 17, 17, 1, 1}, {8, 1024, 1024, 1, 4},
        {8, 8193, 8193, 1, 4}, {65, 17, 17, 1, 1}, {8193, 2, 2, 1, 1},
        {8, 17, 1, 1, 1}, {8, 18, 3, 1, 1}, {8, 17, 17, 2, 1},
    };
    for (const auto & c : div_cases) {
        std::vector<float> left((size_t) c.width * c.rows), right((size_t) c.right_rows * c.stride), observed(left.size());
        for (size_t i = 0; i < left.size(); ++i) { left[i] = ((int) (i % 37) - 18) / 16.0f; }
        for (size_t i = 0; i < right.size(); ++i) { right[i] = 0.5f + (float) (i % 29) / 8.0f; }
        auto * a = allocate(context, left.size() * sizeof(float));
        auto * b = allocate(context, right.size() * sizeof(float));
        auto * out = allocate(context, observed.size() * sizeof(float));
        upload(stream, a, left.data(), left.size() * sizeof(float));
        upload(stream, b, right.data(), right.size() * sizeof(float));
        attrs.config.task_dim = NPUTaskDim3{c.tasks, 1, 1};
        attrs.config.task_mode = c.tasks == 4 ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
        float max_abs = 0;
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            check(test_scalar_div(a, b, out, c.width, c.rows, c.right_rows, c.stride * sizeof(float), &attrs), "scalar divide");
            download(stream, observed.data(), out, observed.size() * sizeof(float));
            for (size_t i = 0; i < left.size(); ++i) {
                const float expected = left[i] / right[(i / c.width % c.right_rows) * c.stride];
                const float error = std::abs(observed[i] - expected);
                if (!std::isfinite(observed[i]) || error > 1e-5f + 1e-5f * std::abs(expected)) {
                    std::fprintf(stderr, "DIV width=%u rows=%u index=%zu actual=%g expected=%g\n", c.width, c.rows, i, observed[i], expected);
                    return 1;
                }
                max_abs = std::max(max_abs, error);
            }
        }
        check(npu_buffer_destroy(a), "free");
        check(npu_buffer_destroy(b), "free");
        check(npu_buffer_destroy(out), "free");
        std::printf("PASS DIV width=%u rows=%u right_rows=%u stride=%u tasks=%u max_abs=%g\n", c.width, c.rows, c.right_rows, c.stride, c.tasks, max_abs);
        std::fflush(stdout);
    }
    check(npu_stream_destroy(stream), "destroy stream");
    check(npu_context_destroy(context), "destroy context");
    check(npu_runtime_close(runtime), "close");
    return 0;
}
#endif
