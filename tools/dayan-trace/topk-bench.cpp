#include <dayan_kernel.h>

#include "ggml-dayan-kernels.h"
#include "ggml-dayan-common.h"
#include "ggml-dayan-memory.h"

#include <stdint.h>

struct alignas(64) Top8Sample {
    uint64_t cycles;
    uint32_t passes;
    uint32_t candidates;
    uint64_t reserved[6];
};

#if defined(__DAYAN_DEVICE__)

#include <stdio.h>

static inline void insert_top8(float value, int32_t index, float * scores, int32_t * ids, uint32_t & count) {
    if (count == 8 && value <= scores[7]) {
        return;
    }
    uint32_t position = count;
    while (position > 0 && value > scores[position - 1]) {
        if (position < 8) {
            scores[position] = scores[position - 1];
            ids[position] = ids[position - 1];
        }
        --position;
    }
    if (position < 8) {
        scores[position] = value;
        ids[position] = index;
    }
    if (count < 8) {
        ++count;
    }
}

static void scalar_top8(const float * input, int32_t * output, uint32_t width, uint32_t rows) {
    for (uint32_t row = 0; row < rows; ++row) {
        const float * values = input + (uint64_t) row * width;
        int32_t * ids = output + (uint64_t) row * width;
        dayan_cache_invalidate(values, width * sizeof(float));
        dayan_cache_invalidate(ids, 8 * sizeof(int32_t));
        float scores[8];
        uint32_t count = 0;
        for (uint32_t index = 0; index < width; ++index) {
            insert_top8(values[index], index, scores, ids, count);
        }
        dayan_cache_flush(ids, 8 * sizeof(int32_t));
    }
}

static void uram_top8(const float * input, int32_t * output, uint32_t width, uint32_t rows) {
    using namespace ggml_dayan_matmul_detail;
    const uint64_t uram = (uint64_t) (uintptr_t) dayan_uram_base();
    const float * values = (const float *) (uintptr_t) uram;
    int32_t * ids = (int32_t *) (uintptr_t) (uram + 16384);
    for (uint32_t row = 0; row < rows; ++row) {
        memory_load_bytes(uram, input + (uint64_t) row * width, width * sizeof(float));
        for (uint32_t index = 0; index < width; ++index) {
            uint32_t position = index < 8 ? index : 8;
            while (position > 0) {
                const int32_t previous = ids[position - 1];
                if (!(values[index] > values[previous])) {
                    break;
                }
                if (position < 8) {
                    ids[position] = previous;
                }
                --position;
            }
            if (position < 8) {
                ids[position] = index;
            }
        }
        memory_store_bytes(output + (uint64_t) row * width, (uint64_t) (uintptr_t) ids, 8 * sizeof(int32_t));
    }
}

static void radix_top8(const float * input, int32_t * output, uint32_t width, uint32_t rows,
                       uint32_t candidate_limit, uint32_t & passes, uint32_t & candidates) {
    using namespace ggml_dayan_vec_detail;
    using ggml_dayan_matmul_detail::memory_load_bytes;
    const uint64_t uram = (uint64_t) (uintptr_t) dayan_uram_base();
    const long histogram = (long) shared;
    const float * values = (const float *) (uintptr_t) uram;
    // Scalar VEC stores need 32-byte alignment, including the 8-byte lane masks.
    const uint64_t masks_address = uram + 16384;
    const uint64_t * masks = (const uint64_t *) (uintptr_t) masks_address;
    const uint64_t prefix_address = uram + 18432;
    const uint32_t * cumulative = (const uint32_t *) (uintptr_t) prefix_address;
    configure_f32_iterators();
    configure_vec_iterator<kVecV3Base>(0, 0, 32, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        memory_load_bytes(uram, input + (uint64_t) row * width, width * sizeof(float));
        uint32_t prefix = 0;
        uint32_t prefix_mask = 0;
        uint32_t remaining = 8;
        uint32_t active = width;
        for (int shift = 26; shift >= 0; shift = shift == 2 ? 0 : shift - 6) {
            const uint32_t digit_mask = shift == 0 ? 3U : 63U;
            configure_vec_grid(1);
            begin_vec_block(2);
            __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
            __tx_vec_lsu_v2sm<0, 2>(histogram);
            start_vec_block(2);
            dayan_wait_vec();

            configure_vec_grid(width / 64);
            begin_vec_block(12);
            // Queue 1: one, shift, digit mask, prefix, prefix mask.
            __tx_vec_lsu_g2v_broadcast32<1, 0>(prefix_mask);
            __tx_vec_lsu_g2v_broadcast32<1, 0>(prefix);
            __tx_vec_lsu_g2v_broadcast32<1, 0>(digit_mask);
            __tx_vec_lsu_g2v_broadcast32<1, 0>(shift);
            __tx_vec_lsu_g2v_broadcast32<1, 0>(1);
            __tx_vec_lsu_u2v<0, 0>((long) uram);
            __tx_vec_vv_int32_bitwise<0, 0, 20, 8>();
            __tx_vec_vv_int32_cmp<0, 19, 0, 2>();
            __tx_vec_vv_int32_srl<0, 1, 17>();
            __tx_vec_vv_int32_bitwise<0, 0, 18, 8>();
            __tx_vec_v_int32_sll<0, 0, 2>();
            __tx_vec_v2sm_sparse_int32_add<0, 16, 0, 0>(histogram);
            start_vec_block(12);
            dayan_wait_vec();
            ++passes;

            configure_vec_grid(1);
            begin_vec_block(29);
            __tx_vec_lsu_g2s<1>((long) 0xfffffffffffffffeULL);
            __tx_vec_lsu_g2s<2>((long) 0xfffffffffffffffcULL);
            __tx_vec_lsu_g2s<3>((long) 0xfffffffffffffff0ULL);
            __tx_vec_lsu_g2s<4>((long) 0xffffffffffffff00ULL);
            __tx_vec_lsu_g2s<5>((long) 0xffffffffffff0000ULL);
            __tx_vec_lsu_g2s<6>((long) 0xffffffff00000000ULL);
            __tx_vec_lsu_sm2v<0, 2>(histogram);
            __tx_vec_shuffle_ror<1, 0>(63);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_vv_mux2<0, 1, 0, 1>();
            __tx_vec_shuffle_ror<1, 0>(62);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_vv_mux2<0, 1, 0, 2>();
            __tx_vec_shuffle_ror<1, 0>(60);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_vv_mux2<0, 1, 0, 3>();
            __tx_vec_shuffle_ror<1, 0>(56);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_vv_mux2<0, 1, 0, 4>();
            __tx_vec_shuffle_ror<1, 0>(48);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_vv_mux2<0, 1, 0, 5>();
            __tx_vec_shuffle_ror<1, 0>(32);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_vv_mux2<0, 1, 0, 6>();
            __tx_vec_lsu_g2v_broadcast32<1, 0>(active - remaining + 1);
            __tx_vec_vv_int32_cmp<0, 16, 0, 3>();
            __tx_vec_lsu_v2u<0, 2>((long) prefix_address);
            __tx_vec_lsu_s2u<0, 2>((long) masks_address);
            start_vec_block(29);
            dayan_wait_vec();
            if (masks[0] == 0) {
                printf("radix pass failed: row=%u shift=%d active=%u remaining=%u prefix=%x cumulative=%u,%u\n",
                       row, shift, active, remaining, prefix, cumulative[0], cumulative[63]);
                dayan_thread_exit(-34);
            }
            const uint32_t bucket = __builtin_ctzll(masks[0]);
            const uint32_t inclusive = cumulative[bucket];
            remaining -= active - inclusive;
            active = inclusive - (bucket == 0 ? 0 : cumulative[bucket - 1]);
            prefix |= bucket << shift;
            prefix_mask |= digit_mask << shift;
            if (8 - remaining + active <= candidate_limit || shift == 0) {
                break;
            }
        }

        configure_vec_grid(width / 64);
        begin_vec_block(4);
        __tx_vec_lsu_g2v_broadcast32<1, 0>(prefix);
        __tx_vec_lsu_u2v<0, 0>((long) uram);
        __tx_vec_vv_fp32_cmp<0, 16, 0, 3>();
        __tx_vec_lsu_s2u<0, 3>((long) masks_address);
        start_vec_block(4);
        dayan_wait_vec();
        int32_t * ids = output + (uint64_t) row * width;
        dayan_cache_invalidate(ids, 8 * sizeof(int32_t));
        float scores[8];
        uint32_t count = 0;
        for (uint32_t chunk = 0; chunk < width / 64; ++chunk) {
            uint64_t mask = masks[chunk * 4];
            for (uint32_t lane = 0; mask != 0; ++lane, mask >>= 1) {
                if (mask & 1) {
                    const uint32_t index = chunk * 64 + lane;
                    insert_top8(values[index], index, scores, ids, count);
                    ++candidates;
                }
            }
        }
        if (count != 8) {
            printf("radix gather failed: row=%u count=%u prefix=%x mask=%llx\n", row, count, prefix,
                   (unsigned long long) masks[0]);
            dayan_thread_exit(-34);
        }
        dayan_cache_flush(ids, 8 * sizeof(int32_t));
    }
}

template <int Mode>
static void run_top8_bench(const float * input, int32_t * output, Top8Sample * samples,
                           uint32_t width, uint32_t rows, uint32_t repetitions, uint32_t candidate_limit) {
    if (width < 64 || width > 4096 || width % 64 != 0 || rows == 0 || repetitions == 0 || candidate_limit < 8) {
        dayan_thread_exit(-22);
    }
    for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
        uint32_t passes = 0;
        uint32_t candidates = 0;
        const uint64_t begin = dayan_get_time();
        if constexpr (Mode == 0) {
            uram_top8(input, output, width, rows);
        } else if constexpr (Mode == 1) {
            scalar_top8(input, output, width, rows);
        } else {
            radix_top8(input, output, width, rows, candidate_limit, passes, candidates);
        }
        const uint64_t elapsed = dayan_get_time() - begin;
        samples[repeat].cycles = elapsed;
        samples[repeat].passes = passes;
        samples[repeat].candidates = candidates;
        dayan_cache_flush(samples + repeat, sizeof(Top8Sample));
    }
}

#endif

__kernel__ void top8_uram(const float * input, int32_t * output, Top8Sample * samples,
                         uint32_t width, uint32_t rows, uint32_t repetitions, uint32_t candidate_limit) {
    run_top8_bench<0>(input, output, samples, width, rows, repetitions, candidate_limit);
}

__kernel__ void top8_scalar(const float * input, int32_t * output, Top8Sample * samples,
                           uint32_t width, uint32_t rows, uint32_t repetitions, uint32_t candidate_limit) {
    run_top8_bench<1>(input, output, samples, width, rows, repetitions, candidate_limit);
}

__kernel__ void top8_radix(const float * input, int32_t * output, Top8Sample * samples,
                          uint32_t width, uint32_t rows, uint32_t repetitions, uint32_t candidate_limit) {
    run_top8_bench<2>(input, output, samples, width, rows, repetitions, candidate_limit);
}

DAYAN_KERNEL_TASK_DESC(top8_uram, 4, 0, 0, 32768, 2, 0, 0, 0);
DAYAN_KERNEL_TASK_DESC(top8_scalar, 4, 0, 0, 0, 0, 0, 0, 0);
DAYAN_KERNEL_TASK_DESC(top8_radix, 4, 0, 256, 32768, 2, 0, 1, 0);

#if !defined(__DAYAN_DEVICE__)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

__host__ static void check(int status, const char * operation) {
    if (status < 0) {
        std::fprintf(stderr, "FAIL: %s status=%d\n", operation, status);
        std::exit(1);
    }
}

__host__ static void wait_event(npu_event_t * event) {
    npu_event_info_t info;
    check(npu_event_wait(event, 300000, &info), "event wait");
    check((int) info.status, "event status");
    check(npu_event_destroy(event), "event destroy");
}

__host__ int main(int argc, char ** argv) {
    if (argc > 7 || (argc > 1 && std::strcmp(argv[1], "--help") == 0)) {
        std::printf("usage: %s [width=256] [rows=1] [repetitions=3] [pattern=softmax] [candidate_limit=32] [mode=compare]\n", argv[0]);
        std::printf("patterns: softmax, ascending, descending, ties, or an FP32 input file; width multiple of 64\n");
        std::printf("modes: all, compare, scalar-direct, vec-radix, uram-insertion\n");
        return argc > 7 ? 2 : 0;
    }
    const uint32_t width = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 256;
    const uint32_t rows = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 1;
    const uint32_t repetitions = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 3;
    const char * pattern = argc > 4 ? argv[4] : "softmax";
    const uint32_t candidate_limit = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 32;
    const char * selected_mode = argc > 6 ? argv[6] : "compare";
    if (width < 64 || width > 4096 || width % 64 != 0 || rows == 0 || rows > 256 ||
        repetitions < 2 || repetitions > 100 || candidate_limit < 8 || candidate_limit > width ||
        (std::strcmp(selected_mode, "all") != 0 && std::strcmp(selected_mode, "compare") != 0 &&
         std::strcmp(selected_mode, "scalar-direct") != 0 &&
         std::strcmp(selected_mode, "vec-radix") != 0 && std::strcmp(selected_mode, "uram-insertion") != 0)) {
        return 2;
    }
    std::vector<float> input((size_t) width * rows);
    const bool synthetic = std::strcmp(pattern, "softmax") == 0 || std::strcmp(pattern, "ascending") == 0 ||
                           std::strcmp(pattern, "descending") == 0 || std::strcmp(pattern, "ties") == 0;
    if (!synthetic) {
        FILE * file = std::fopen(pattern, "rb");
        if (file == nullptr) {
            std::perror(pattern);
            return 2;
        }
        const size_t count = std::fread(input.data(), sizeof(float), input.size(), file);
        std::fclose(file);
        if (count != input.size() || std::any_of(input.begin(), input.end(), [](float value) {
                return !std::isfinite(value) || std::signbit(value);
            })) {
            std::fprintf(stderr, "input must contain width * rows finite nonnegative FP32 values without negative zero\n");
            return 2;
        }
    }
    uint32_t random = 0x31415926;
    for (uint32_t row = 0; synthetic && row < rows; ++row) {
        double sum = 0;
        for (uint32_t index = 0; index < width; ++index) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            float value;
            if (std::strcmp(pattern, "softmax") == 0) {
                value = std::exp((float) (random >> 8) / 16777216.0f * 8.0f - 4.0f);
            } else if (std::strcmp(pattern, "ascending") == 0) {
                value = (float) (index + 1);
            } else if (std::strcmp(pattern, "descending") == 0) {
                value = (float) (width - index);
            } else if (std::strcmp(pattern, "ties") == 0) {
                value = 1.0f;
            } else {
                return 2;
            }
            input[(size_t) row * width + index] = value;
            sum += value;
        }
        for (uint32_t index = 0; index < width; ++index) {
            input[(size_t) row * width + index] /= (float) sum;
        }
    }
    std::vector<int32_t> expected((size_t) rows * 8);
    std::vector<int32_t> ids(width);
    for (uint32_t row = 0; row < rows; ++row) {
        std::iota(ids.begin(), ids.end(), 0);
        std::stable_sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
            return input[(size_t) row * width + a] > input[(size_t) row * width + b];
        });
        std::copy_n(ids.begin(), 8, expected.begin() + row * 8);
    }

    npu_runtime_options_t options;
    npu_runtime_options_init(&options);
    const char * socket = std::getenv("DAYAN_SERVER_SOCKET");
    if (socket && *socket) {
        options.server_socket = socket;
    }
    npu_runtime_t * runtime = nullptr;
    npu_context_t * context = nullptr;
    npu_stream_t * stream = nullptr;
    check(npu_runtime_open(&runtime, &options), "runtime open");
    check(npu_context_create(runtime, nullptr, &context), "context create");
    check(npu_stream_create(context, nullptr, &stream), "stream create");
    npu_buffer_t * buffers[3] = {};
    const size_t sizes[] = { input.size() * sizeof(float), input.size() * sizeof(int32_t),
                             repetitions * sizeof(Top8Sample) };
    for (int index = 0; index < 3; ++index) {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, sizes[index]);
        desc.alignment = 64;
        check(npu_buffer_create(context, &desc, &buffers[index]), "buffer create");
    }
    npu_event_t * event = nullptr;
    check(npu_memcpy_h2d_async(stream, buffers[0], 0, input.data(), sizes[0], &event), "upload input");
    wait_event(event);
    using Kernel = int (*)(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, uint32_t, uint32_t, uint32_t,
                           uint32_t, const npu_kernel_launch_attrs_t *);
    const Kernel kernels[] = { top8_scalar, top8_radix, top8_uram };
    const char * names[] = { "scalar-direct", "vec-radix", "uram-insertion" };
    std::vector<int32_t> output(input.size(), -1);
    std::vector<Top8Sample> samples(repetitions);
    bool passed = true;
    for (int mode = 0; mode < 3; ++mode) {
        const bool selected = std::strcmp(selected_mode, "all") == 0 || std::strcmp(selected_mode, names[mode]) == 0 ||
                              (std::strcmp(selected_mode, "compare") == 0 && mode < 2);
        if (!selected) {
            continue;
        }
        std::fill(output.begin(), output.end(), -1);
        check(npu_memcpy_h2d_async(stream, buffers[1], 0, output.data(), sizes[1], &event), "clear output");
        wait_event(event);
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, context, stream, names[mode]);
        attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = 300000;
        check(kernels[mode](buffers[0], buffers[1], buffers[2], width, rows, repetitions, candidate_limit, &attrs), "kernel");
        check(npu_memcpy_d2h_async(stream, output.data(), buffers[1], 0, sizes[1], &event), "download output");
        wait_event(event);
        check(npu_memcpy_d2h_async(stream, samples.data(), buffers[2], 0, sizes[2], &event), "download samples");
        wait_event(event);
        bool valid = true;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t index = 0; index < width; ++index) {
                const int32_t wanted = index < 8 ? expected[row * 8 + index] : -1;
                if (output[(size_t) row * width + index] != wanted) {
                    std::fprintf(stderr, "%s row=%u index=%u expected=%d actual=%d\n", names[mode], row, index,
                                 wanted, output[(size_t) row * width + index]);
                    valid = false;
                    break;
                }
            }
        }
        passed = passed && valid;
        std::vector<uint64_t> cycles;
        for (uint32_t repeat = 0; repeat < repetitions; ++repeat) {
            std::printf("sample mode=%s width=%u rows=%u pattern=%s repeat=%u cycles=%llu passes=%u candidates=%u valid=%d\n",
                        names[mode], width, rows, pattern, repeat, (unsigned long long) samples[repeat].cycles,
                        samples[repeat].passes, samples[repeat].candidates, valid);
            if (repeat != 0) {
                cycles.push_back(samples[repeat].cycles);
            }
        }
        std::sort(cycles.begin(), cycles.end());
        std::printf("result mode=%s width=%u rows=%u pattern=%s median_cycles=%llu cycles_per_row=%.2f valid=%d\n",
                    names[mode], width, rows, pattern, (unsigned long long) cycles[cycles.size() / 2],
                    (double) cycles[cycles.size() / 2] / rows, valid);
        std::fflush(stdout);
    }
    for (npu_buffer_t * buffer : buffers) {
        check(npu_buffer_destroy(buffer), "buffer destroy");
    }
    check(npu_stream_destroy(stream), "stream destroy");
    check(npu_context_destroy(context), "context destroy");
    check(npu_runtime_close(runtime), "runtime close");
    std::printf("[host] %s: Top-8 indices and output guards\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

#endif
