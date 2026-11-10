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

#include "ggml-dayan-matmul.h"
#include "ggml-dayan-mxfp4-layout.h"
#include "ggml-dayan-workers.h"

#if defined(__DAYAN_DEVICE__)
namespace ggml_dayan_mxfp4_detail {

using namespace ggml_dayan_matmul_detail;

// VisualSwizzle: peak 1; the 320B mapping needs 1024B-aligned buffer boundaries.
static uint64_t row_swizzle(int stride) {
    if (stride == 320) { return MAT_BATCH_SWIZZLE(9, 1, 1); }
    int bit = 0;
    while ((stride & 1) == 0) { ++bit; stride >>= 1; }
    return bit <= 7 ? 0 : MAT_BATCH_SWIZZLE(7, bit < 10 ? bit - 7 : 3, bit < 10 ? 3 : bit - 7);
}

template <int Operation>
static inline void emit_post_activation(uint64_t gate, uint64_t up) {
    using namespace ggml_dayan_vec_detail;
    if constexpr (Operation == 2) { emit_swiglu(gate, up); }
    else if constexpr (Operation == 3) { emit_geglu(gate, up); }
    else {
        __tx_vec_lsu_u2v<0, 0>(gate);
        if constexpr (Operation == 1) {
            __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(-1.4426950408889634f));
            __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
            __tx_vec_v_fp32_sfu<0, 0, 1>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(1.0f));
            __tx_vec_vv_fp32_add<0, 16, 0, 0>();
            __tx_vec_v_fp32_sfu<0, 0, 5>();
            __tx_vec_vv_fp32_mul<0, 0, 4, 0>();
        }
    }
}

template <int Operation>
static void post_values(uint64_t gate, uint64_t up, int rows, int columns, int stride, uint32_t output_type) {
    using namespace ggml_dayan_vec_detail;
    constexpr int activation_instructions = Operation == 3 ? 26 : Operation == 2 ? 10 : Operation == 1 ? 8 : 1;
    constexpr int saved_bf16 = 16 + (Operation == 3 ? 10 : Operation == 2 ? 3 : Operation == 1 ? 2 : 0);
    const bool mx = output_type == kDayanDataTypeMXFP4;
    const bool f32 = output_type == kDayanDataTypeF32;
    const int block_bytes = mx ? 512 : 256;
    const long instructions = mx ? 2 * activation_instructions + 8 : activation_instructions + (f32 ? 1 : 6);
    configure_vec_grid(columns * 4 / block_bytes, rows);
    configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, block_bytes, stride);
    configure_vec_iterator<ggml_dayan_vec_detail::kVecV1Base>(0, 0, block_bytes, stride);
    configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, block_bytes, stride, mx ? 64 : 32);
    begin_vec_block(instructions);
    emit_post_activation<Operation>(gate, up);
    if (mx) {
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
        // Queue 1 keeps the first half across the second activation (at most ten queue writes).
        __tx_vec_shuffle_compress<1, 0>(0);
        emit_post_activation<Operation>(gate + 256, up + 256);
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
        __tx_vec_shuffle_compress<0, 0>(0);
        __tx_vec_lsu_g2s<0>((long) 0xffffffff00000000ULL);
        __tx_vec_vv_mux2<0, saved_bf16, 0, 0>();
        __tx_vec_cvt_mxfp4_bf16<0, 0, 0>();
    } else if (!f32) {
        if (output_type == kDayanDataTypeBF16) { __tx_vec_cvt_bf16_fp32<0, 0, 0>(); }
        else { __tx_vec_cvt_fp16_fp32<0, 0, 0>(); }
        __tx_vec_shuffle_compress<0, 0>(0);
        __tx_vec_shuffle_xor<1, 0>(16);
        __tx_vec_lsu_g2s<0>((long) 0xffffffff00000000ULL);
        __tx_vec_vv_mux2<0, 0, 16, 0>();
    }
    // Read the entire physical block before overwriting it; 128B group positions stay unchanged.
    __tx_vec_lsu_v2u<0, 3>(gate);
    start_vec_block(instructions);
}

static void quantize_lowp(uint64_t input, int rows, int columns, int stride, bool fp16) {
    using namespace ggml_dayan_vec_detail;
    const long instructions = fp16 ? 34 : 3;
    configure_vec_grid(columns / 128, rows);
    configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, 256, stride);
    configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, 256, stride);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    if (fp16) {
        swap_vec_lane_bits<0x0000ffffffff0000ULL, 48>();
        swap_vec_lane_bits<0x00ffff0000ffff00ULL, 24>();
        swap_vec_lane_bits<0x0ff00ff00ff00ff0ULL, 12>();
        swap_vec_lane_bits<0x3c3c3c3c3c3c3c3cULL, 6>();
        swap_vec_lane_bits<0x6666666666666666ULL, 3>();
        __tx_vec_shuffle_xor<1, 0>(1);
        __tx_vec_lsu_g2s<0>((long) 0xaaaaaaaaaaaaaaaaULL);
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_v_int32_srl<1, 0, 16>();
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_cvt_fp32_fp16<0, 0, 0>();
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
        __tx_vec_shuffle_compress<1, 0>(0);
        // Keep the first packed half in queue 1 while expanding the second half.
        __tx_vec_vv_mux2<0, 18, 4, 0>();
        __tx_vec_v_int32_srl<1, 0, 16>();
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_cvt_fp32_fp16<0, 0, 0>();
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
        __tx_vec_shuffle_compress<0, 0>(0);
        __tx_vec_lsu_g2s<0>((long) 0xffffffff00000000ULL);
        __tx_vec_vv_mux2<0, 17, 0, 0>();
    }
    __tx_vec_cvt_mxfp4_bf16<0, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(input);
    start_vec_block(instructions);
}

static void pack_quantized(uint64_t input, uint64_t packed, uint64_t scales, int rows, int columns,
                           int input_stride, int group_stride, int packed_stride, int scale_stride, uint64_t swizzle) {
    long checksum = 0;
    set_lsu_matrix<0>(checksum, rows, 64, input_stride, packed_stride);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(swizzle);
    __tx_lsu_tcsr_write<LSU_DIM0>(columns / 32 - 1);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(group_stride);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(16);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(15);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(15);
    __tx_lsu_dma_u2u<0>(packed, input);
    set_lsu_matrix<0>(checksum, (rows + 7) & ~7, 64, input_stride, 4);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(swizzle);
    __tx_lsu_tcsr_write<LSU_DIM0>(3);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(group_stride);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(1);
    __tx_lsu_tcsr_write<LSU_DIM2>(columns / 128 - 1);
    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(group_stride * 4);
    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(scale_stride);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(0);
    __tx_lsu_dma_u2u<0>(scales, input + group_stride / 2);
}

static void store_quantized(uint64_t packed, uint64_t scales, uint64_t output, uint64_t output_scales,
                            int rows, int columns, int packed_stride, int scale_stride,
                            uint64_t output_stride, uint64_t output_scale_stride) {
    long checksum = 0;
    set_lsu_matrix<0>(checksum, rows, columns / 2, packed_stride, output_stride);
    __tx_lsu_dma_u2d<0>(output, packed);
    const int scale_bytes = rows * 4;
    const int padded = (scale_bytes + 63) & ~63;
    set_lsu_matrix<0>(checksum, columns / 128, padded, scale_stride, output_scale_stride);
    store_matrix<0>(scale_bytes, columns / 128, padded, columns / 128, scales, output_scales);
}

static void convert(const DayanMxfp4Config * cfg, const void * input, void * output, void * scales, const NativeContext & context = {}) {
    if (!cfg) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {(uint64_t) cfg->m * cfg->n, ((cfg->m + 127) / 128) * ((cfg->n + 511) / 512), 320U * 1024U, 1}, [&](const NativeContext & worker) {
            convert(cfg, input, output, scales, worker);
        })) { return; }
    using namespace ggml_dayan_vec_detail;
    const uint64_t base = native_uram(context);
    enum : int { Rows = 128, Columns = 512 };
    const uint64_t nt = (cfg->n + Columns - 1) / Columns;
    const int bytes = cfg->source_type == kDayanDataTypeF32 ? 4 : 2;
    const int value_stride = Columns * bytes;
    const uint64_t packed = base + Rows * value_stride, scale = packed + Rows * Columns / 2;
    DAYAN_PARAMS_IF(!base || scale + Rows * Columns / 32 + 64 - base > native_uram_size(context)) {
        DAYAN_PARAMS_ERROR();
    }
    DAYAN_PARAMS_IF((base | packed | scale) & 31) {
        DAYAN_PARAMS_ERROR();
    }
    long checksum = 0;
    for (uint64_t tile = context.task; tile < ((cfg->m + Rows - 1) / Rows) * nt; tile += context.tasks) {
        const uint64_t row = tile / nt * Rows, col = tile % nt * Columns;
        const int rows = extent(cfg->m - row, Rows), cols = extent(cfg->n - col, Columns);
        const int columns = (cols + 127) & ~127;
        set_lsu_matrix<0>(checksum, rows, Columns * bytes, cfg->a_stride, Columns * bytes);
        load_matrix<0>(cols * bytes, rows, Columns * bytes, rows, base,
                       (uintptr_t) input + row * cfg->a_stride + col * bytes);
        dayan_wait_lsu();
        if (bytes == 2) {
            quantize_lowp(base, rows, columns, value_stride, cfg->source_type == kDayanDataTypeF16);
        } else {
            post_values<0>(base, 0, rows, columns, value_stride, kDayanDataTypeMXFP4);
        }
        dayan_wait_vec();
        pack_quantized(base, packed, scale, rows, columns, value_stride, bytes * 32, Columns / 2, Rows * 4, 0);
        dayan_wait_lsu();
        store_quantized(packed, scale, (uintptr_t) output + row * cfg->output_stride + col / 2,
                        (uintptr_t) scales + col / 128 * cfg->output_scale_stride + row * 4,
                        rows, columns, Columns / 2, Rows * 4, cfg->output_stride, cfg->output_scale_stride);
        dayan_wait_lsu();
    }
}

struct Phases {
    uint64_t mat = 0;
    uint64_t post = 0;
};

// Shared 0..11 belong to the other matrix and attention pipelines.
static constexpr int shared_resource_begin = 12;

struct State {
    const DayanMxfp4Config * cfg;
    Phases * phases;
    DayanMxfp4Tile shape;
    uint64_t input[3], scales[3], output, output_scales;
    uint64_t physical, local, packed, packed_scales, peer_uram[4];
    uint64_t a[2], as[2], b[2], bs[2], c[DayanMxfp4Tile::max_c_buffers];
    int kt, nt, grid_rows, grid_cols;
    uint64_t tiles, task, tasks, load_phases[2];
    bool share;
};

struct Tile { uint64_t row, col; int rows, cols, users; };

template <int M, int N>
static Tile tile_at(const State & s, uint64_t out, int slice, int lane) {
    constexpr int slice_n = N / 2;
    if (M == 32 && s.cfg->m == 1 && s.tasks == 1) {
        const uint64_t col = out * N;
        const int cols = extent((int64_t) s.cfg->n - col, N);
        return {0, col + slice * slice_n, 1, extent(cols - slice * slice_n, slice_n),
                (cols + slice_n - 1) / slice_n};
    }
    const uint64_t row_base = out / s.nt * s.grid_rows * M;
    const uint64_t col_base = out % s.nt * s.grid_cols * N;
    const int group_rows = extent((int64_t) s.cfg->m - row_base, s.grid_rows * M);
    const int group_cols = extent((int64_t) s.cfg->n - col_base, s.grid_cols * N);
    const int row_step = ((group_rows + s.grid_rows - 1) / s.grid_rows + 31) & ~31;
    const int alignment = s.cfg->output_type == kDayanDataTypeMXFP4 ? 128 : 16;
    const int col_step = ((group_cols + s.grid_cols - 1) / s.grid_cols + alignment - 1) & ~(alignment - 1);
    const uint64_t row = row_base + lane / s.grid_cols * row_step;
    const uint64_t col = col_base + lane % s.grid_cols * col_step;
    const int cols = extent((int64_t) s.cfg->n - col, col_step);
    return {row, col + slice * slice_n, extent((int64_t) s.cfg->m - row, row_step),
            extent(cols - slice * slice_n, slice_n), (cols + slice_n - 1) / slice_n};
}

template <int M, int N, bool B>
static void load(void * arg) {
    constexpr int slice_n = N / 2;
    State & s = *(State *) arg;
    const auto & shape = s.shape;
    const bool share = s.share && (!B || s.grid_rows == 2);
    const bool broadcast = !B && s.grid_rows == 1;
    const int peer = broadcast ? 0 : s.task ^ (B ? 2 : 1);
    const int pair = broadcast ? 0 : B ? s.task & 1 : s.task >> 1;
    uint64_t phases = 0;
    long checksum = 0;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int step = 0; step < s.kt; ++step) {
            for (int slice = 0; slice < (B ? 2 : 1); ++slice) {
                const Tile tile = tile_at<M, N>(s, out, slice, s.task);
                const Tile other = tile_at<M, N>(s, out, slice, share ? peer : s.task);
                const bool active = tile.rows && tile.cols;
                if (!active && !(share && other.rows && other.cols)) { continue; }
                const int slot = B ? slice : (out * s.kt + step) & 1;
                const int rows = B ? tile.cols : tile.rows;
                const int capacity = B ? slice_n : M;
                const uint64_t first = B ? tile.col : tile.row;
                const auto resource = DAYAN_LOCAL_RESOURCE((B ? 2 : 0) + slot);
                const auto local = active ? DAYAN_RESOURCE_MASK(resource) : 0;
                const auto ready = DAYAN_SHARED_RESOURCE(shared_resource_begin + (B ? 0 : 4) + pair * 2 + slot);
                const auto ready_mask = DAYAN_RESOURCE_MASK(ready);
                int role = DAYAN_RESOURCE_ROLE_PRODUCER;
                if (share && broadcast) {
                    matmul_check(dayan_resource_wait(local | (s.task == 0 ? ready_mask : 0), s.task == 0 ? 0 : ready_mask, phases, 0));
                    role = s.task == 0 ? DAYAN_RESOURCE_ROLE_PRODUCER : DAYAN_RESOURCE_ROLE_CONSUMER;
                } else if (share) {
                    role = dayan_resource_wait(active ? local | ready_mask : 0, ready_mask, phases, 0);
                    matmul_check(role);
                } else {
                    matmul_check(dayan_resource_wait_free(resource));
                }
                const int users = (B ? 1 : tile.users) * shape.products;
                if (!active) {
                    matmul_check(dayan_resource_consume_lsu(ready_mask));
                } else if (role == DAYAN_RESOURCE_ROLE_PRODUCER) {
                    for (int product = 0; product < (B ? shape.products : 1); ++product) {
                        const int operand = B ? 1 + product : 0;
                        const uint64_t destination = (B ? s.b[slot] : s.a[slot]) + product * capacity * shape.input_stride;
                        const uint64_t source_stride = B ? s.cfg->b_stride : s.cfg->a_stride;
                        const int inner = extent(s.cfg->k - step * shape.k, shape.k);
                        const int bytes = shape.scales ? inner / 2 : inner * 2;
                        set_lsu_matrix<0>(checksum, rows, shape.input_stride, source_stride, shape.input_stride);
                        __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(row_swizzle(shape.input_stride));
                        load_matrix<0>(bytes, rows, shape.input_stride, rows, destination,
                                       s.input[operand] + first * source_stride + step * shape.input_stride);
                        if (shape.scales) {
                            const uint64_t scale_stride = B ? s.cfg->b_scale_stride : s.cfg->a_scale_stride;
                            // LSU zero-fills a 64B tail; keep it out of the next scale plane.
                            const uint64_t local_stride = B ? capacity * 4 : shape.a_scale_stride();
                            const uint64_t target = (B ? s.bs[slot] : s.as[slot]) + product * (shape.k / 128) * local_stride;
                            const int padded = (rows + 7) & ~7;
                            set_lsu_matrix<0>(checksum, (inner + 127) / 128, ((padded * 4 + 63) & ~63), scale_stride, local_stride);
                            load_matrix<0>(padded * 4, (inner + 127) / 128, ((padded * 4 + 63) & ~63), (inner + 127) / 128,
                                           target, s.scales[operand] + (uint64_t) step * (shape.k / 128) * scale_stride + first * 4);
                        }
                    }
                    matmul_check(dayan_resource_produce_lsu(resource, users, 0));
                    if (share) { matmul_check(dayan_resource_produce_lsu(ready, broadcast ? 3 : 1, 0)); }
                } else {
                    for (int product = 0; product < (B ? shape.products : 1); ++product) {
                        const uint64_t destination = (B ? s.b[slot] : s.a[slot]) + product * capacity * shape.input_stride;
                        const uint64_t bytes = capacity * shape.input_stride;
                        // Copy the full allocation: the 320B swizzle can move tail data between rows.
                        set_lsu_matrix<0>(checksum, 1, bytes, bytes, bytes);
                        load_matrix<0>(bytes, 1, bytes, 1, destination, s.peer_uram[peer] + destination - s.local);
                        if (shape.scales) {
                            const uint64_t stride = B ? capacity * 4 : shape.a_scale_stride();
                            const uint64_t scale_bytes = (shape.k / 128) * stride;
                            const uint64_t target = (B ? s.bs[slot] : s.as[slot]) + product * scale_bytes;
                            set_lsu_matrix<0>(checksum, 1, scale_bytes, scale_bytes, scale_bytes);
                            load_matrix<0>(scale_bytes, 1, scale_bytes, 1, target, s.peer_uram[peer] + target - s.local);
                        }
                    }
                    matmul_check(dayan_resource_produce_lsu(resource, users, ready_mask));
                }
                if (share) { phases ^= ready_mask; }
            }
        }
    }
    dayan_wait_lsu();
    s.load_phases[B] = phases;
}

template <int M, int N>
static void calculate(State & s) {
    constexpr int slice_n = N / 2, c_stride = N * 2;
    const auto & shape = s.shape;
    uint64_t & phases = s.phases->mat;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(shape.input_stride);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(shape.input_stride);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(c_stride);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(row_swizzle(shape.input_stride));
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(row_swizzle(shape.input_stride));
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(row_swizzle(c_stride));
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_ASCALESTRIDE>(shape.a_scale_stride());
    __tx_mat_tcsr_write<MAT_BSCALESTRIDE>(slice_n * 4);
    const long datatype = s.cfg->weight_type == kDayanDataTypeMXFP4 ? MAT_DATATYPE_PAIR(0x38, 0x38) :
        s.cfg->weight_type == kDayanDataTypeBF16 ? BaseNPUInstBuilder::DT_BF16 : BaseNPUInstBuilder::DT_FP16;
    __tx_mat_tcsr_write<MAT_DATATYPE>(datatype);
    uint64_t completed = 0;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        const Tile first = tile_at<M, N>(s, out, 0, s.task);
        if (!first.rows || !first.cols) { continue; }
        for (int step = 0; step < s.kt; ++step) {
            const int slot = (out * s.kt + step) & 1;
            for (int slice = 0; slice < 2; ++slice) {
                const Tile tile = tile_at<M, N>(s, out, slice, s.task);
                if (!tile.rows || !tile.cols) { continue; }
                const int c_slot = (completed + slice) % s.shape.c_buffers;
                const auto result = DAYAN_LOCAL_RESOURCE(4 + c_slot);
                const auto a = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(slot));
                const auto b = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(2 + slice));
                matmul_check(dayan_resource_wait(step == 0 ? DAYAN_RESOURCE_MASK(result) : 0, a | b, phases, 0));
                __tx_mat_tcsr_write<MAT_M>(tile.rows - 1);
                __tx_mat_tcsr_write<MAT_N>(tile.cols - 1);
                __tx_mat_tcsr_write<MAT_K>(extent(s.cfg->k - step * shape.k, shape.k) - 1);
                for (int product = 0; product < shape.products; ++product) {
                    const uint64_t weight = s.b[slice] + product * slice_n * shape.input_stride;
                    const uint64_t c = s.c[c_slot] + product * M * c_stride;
                    __tx_mat_setptr<0>(0, s.as[slot], s.bs[slice] + product * (shape.k / 128) * slice_n * 4);
                    if (step == 0) { __tx_mat_mm<0>(c, s.a[slot], weight); }
                    else { __tx_mat_mma<0>(c, s.a[slot], weight); }
                    if (step + 1 == s.kt && product + 1 == shape.products) {
                        matmul_check(dayan_resource_produce_mat(result, 1, a | b));
                    } else {
                        matmul_check(dayan_resource_consume_mat(a | b));
                    }
                }
                phases ^= b;
                if (slice + 1 == tile.users) { phases ^= a; }
            }
        }
        completed += first.users;
    }
    dayan_wait_mat();
}

template <int M, int N>
static void post(void * arg) {
    constexpr int slice_n = N / 2, c_stride = N * 2;
    State & s = *(State *) arg;
    using namespace ggml_dayan_vec_detail;
    uint64_t & phases = s.phases->post;
    long checksum = 0;
    uint64_t completed = 0;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        const Tile first = tile_at<M, N>(s, out, 0, s.task);
        if (!first.rows || !first.cols) { continue; }
        for (int slice = 0; slice < 2; ++slice) {
            const Tile tile = tile_at<M, N>(s, out, slice, s.task);
            if (!tile.rows || !tile.cols) { continue; }
            const int c_slot = (completed + slice) % s.shape.c_buffers;
            const auto mask = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(4 + c_slot));
            matmul_check(dayan_resource_wait(0, mask, phases, 0));
            const uint64_t c = s.physical + s.c[c_slot] - s.local;
            const uint64_t destination = s.output + tile.row * s.cfg->output_stride;
            if (s.shape.post) {
                const uint64_t up = c + M * c_stride;
                // C swizzles permute 128B groups within each row, preserving K32 quantization groups.
                switch (s.cfg->operation) {
                    case 1: post_values<1>(c, up, tile.rows, slice_n, c_stride, s.cfg->output_type); break;
                    case 2: post_values<2>(c, up, tile.rows, slice_n, c_stride, s.cfg->output_type); break;
                    case 3: post_values<3>(c, up, tile.rows, slice_n, c_stride, s.cfg->output_type); break;
                    default: post_values<0>(c, up, tile.rows, slice_n, c_stride, s.cfg->output_type); break;
                }
                dayan_wait_vec();
            }
            if (s.cfg->output_type == kDayanDataTypeMXFP4) {
                const uint64_t packed = s.shape.products == 2 ? c + M * c_stride : s.packed;
                const uint64_t packed_scales = packed + M * slice_n / 2;
                // The up accumulator is dead after activation and can hold the packed result.
                if (s.shape.products == 1) { dayan_wait_lsu(); }
                const int columns = (tile.cols + 127) & ~127;
                pack_quantized(c, packed, packed_scales, tile.rows, columns, c_stride, 128,
                               slice_n / 2, M * 4, row_swizzle(c_stride));
                dayan_wait_lsu();
                store_quantized(packed, packed_scales, destination + tile.col / 2,
                                s.output_scales + tile.col / 128 * s.cfg->output_scale_stride + tile.row * 4,
                                tile.rows, columns, slice_n / 2, M * 4,
                                s.cfg->output_stride, s.cfg->output_scale_stride);
            } else if (s.cfg->output_type == kDayanDataTypeF32) {
                set_lsu_matrix<0>(checksum, tile.rows, c_stride, c_stride, s.cfg->output_stride);
                __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(row_swizzle(c_stride));
                store_matrix<0>(tile.cols * 4, tile.rows, c_stride, tile.rows, c, destination + tile.col * 4);
            } else {
                set_lsu_matrix<0>(checksum, tile.rows, 64, c_stride, s.cfg->output_stride);
                __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(row_swizzle(c_stride));
                __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(128);
                if (tile.cols >= 32) {
                    __tx_lsu_tcsr_write<LSU_DIM0>(tile.cols / 32 - 1);
                    __tx_lsu_dma_u2d<0>(destination + tile.col * 2, c);
                }
                if (tile.cols % 32) {
                    __tx_lsu_tcsr_write<LSU_DIM0>(0);
                    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>((tile.cols % 32) * 2 - 1);
                    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>((tile.cols % 32) * 2 - 1);
                    __tx_lsu_dma_u2d<0>(destination + (tile.col + tile.cols / 32 * 32) * 2, c + tile.cols / 32 * 128);
                }
            }
            matmul_check(dayan_resource_consume_lsu(mask));
            phases ^= mask;
        }
        completed += first.users;
    }
    dayan_wait_lsu();
}

template <int M, int N>
static int run_tiled(State & s) {
    constexpr int slice_n = N / 2, c_stride = N * 2;
    const auto * cfg = s.cfg;
    const auto & shape = s.shape;
    s.kt = (cfg->k + shape.k - 1) / shape.k;
    s.nt = (cfg->n + s.grid_cols * N - 1) / (s.grid_cols * N);
    s.tiles = (cfg->m + s.grid_rows * M - 1) / (s.grid_rows * M) * s.nt;
    s.physical = (uintptr_t) dayan_uram_base();
    s.local = s.physical & (Constants::kUramWindowStride - 1);
    uint64_t address = s.local;
    for (int i = 0; i < 2; ++i) {
        s.a[i] = address; address += shape.m * shape.input_stride;
        s.b[i] = address; address += shape.products * slice_n * shape.input_stride;
    }
    address = s.local + shape.accumulator_offset();
    for (int i = 0; i < shape.c_buffers; ++i) {
        s.c[i] = address; address += shape.products * shape.m * c_stride;
    }
    for (int i = 0; i < 2; ++i) {
        s.as[i] = address; address += shape.scales ? shape.a_scale_stride() * (shape.k / 128) : 0;
        s.bs[i] = address; address += shape.scales ? shape.products * slice_n * (shape.k / 32) : 0;
    }
    s.packed = s.physical + shape.scratch_offset();
    s.packed_scales = s.packed + (shape.post && shape.products == 1 ? shape.m * slice_n / 2 : 0);
    DAYAN_PARAMS_IF(!s.physical || shape.bytes() > dayan_uram_size() || shape.bytes() > Constants::kUramWindowStride - s.local) { return -12; }
    DAYAN_PARAMS_IF((s.packed | s.packed_scales) & 31) { return -22; }
    for (int i = 0; i < 2; ++i) {
        DAYAN_PARAMS_IF(((s.a[i] | s.b[i]) & 1023) || ((s.as[i] | s.bs[i]) & 31)) { return -22; }
    }
    for (int i = 0; i < shape.c_buffers; ++i) {
        DAYAN_PARAMS_IF(s.c[i] & 1023) { return -22; }
    }
    if (s.share) {
        auto & cluster = dayan_cluster_state(cfg->cluster_workspace);
        cluster.core[s.task].uram = s.physical;
        dayan_cluster_flush(cfg->cluster_workspace, s.task);
        matmul_check(dayan_cluster_sync());
        for (int peer = 0; peer < 4; ++peer) {
            if (peer != (int) s.task) { dayan_cluster_invalidate(cfg->cluster_workspace, peer); }
            s.peer_uram[peer] = cluster.core[peer].uram;
        }
        matmul_check(dayan_cluster_sync());
    }
    if constexpr (M == 8) {
        load<M, N, false>(&s);
        load<M, N, true>(&s);
        calculate<M, N>(s);
        post<M, N>(&s);
        return 0;
    }
    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = 1;
    resources.local_resource_count = 4 + shape.c_buffers;
    resources.shared_resource_count = s.share ? shared_resource_begin + 8 : 0;
    matmul_check(dayan_create_thread("mxfp4-load-a", load<M, N, false>, &s, &resources));
    matmul_check(dayan_create_thread("mxfp4-load-b", load<M, N, true>, &s, &resources));
    resources.vec_tcsr_count = 1;
    matmul_check(dayan_create_thread("mxfp4-post", post<M, N>, &s, &resources));
    calculate<M, N>(s);
    matmul_check(dayan_thread_join_all());
    if (s.share) {
        matmul_check(dayan_cluster_sync());
        // Return the dedicated Shared tokens to phase zero for the next mega slot or expert.
        for (int operand = 0; operand < 2; ++operand) {
            const bool owner = operand ? s.grid_rows == 2 && s.task < 2 : s.grid_rows == 1 ? s.task == 0 : (s.task & 1) == 0;
            if (!owner) { continue; }
            const int pair = operand ? s.task & 1 : s.grid_rows == 1 ? 0 : s.task >> 1;
            for (int slot = 0; slot < 2; ++slot) {
                const auto ready = DAYAN_SHARED_RESOURCE(shared_resource_begin + (operand ? 0 : 4) + pair * 2 + slot);
                const auto mask = DAYAN_RESOURCE_MASK(ready);
                if (!(s.load_phases[operand] & mask)) { continue; }
                matmul_check(dayan_resource_wait_free(ready));
                matmul_check(dayan_resource_produce_lsu(ready, 1, 0));
                matmul_check(dayan_resource_wait_ready(ready, 1));
                matmul_check(dayan_resource_consume_lsu(mask));
            }
        }
        dayan_wait_lsu();
        return dayan_cluster_sync();
    }
    return 0;
}

static int run(const DayanMxfp4Config * cfg, const void * a, const void * as, const void * b, const void * bs,
               const void * up, const void * ups, void * output, void * output_scales, Phases * phases = nullptr) {
    Phases local_phases;
    State s = {};
    s.cfg = cfg;
    s.phases = phases ? phases : &local_phases;
    s.input[0] = (uintptr_t) a; s.input[1] = (uintptr_t) b; s.input[2] = (uintptr_t) up;
    s.scales[0] = (uintptr_t) as; s.scales[1] = (uintptr_t) bs; s.scales[2] = (uintptr_t) ups;
    s.output = (uintptr_t) output; s.output_scales = (uintptr_t) output_scales;
    s.tasks = (uint64_t) dayan_task_dim(DAYAN_TASK_X) * dayan_task_dim(DAYAN_TASK_Y);
    s.task = dayan_task_id(DAYAN_TASK_X) + (uint64_t) dayan_task_dim(DAYAN_TASK_X) * dayan_task_id(DAYAN_TASK_Y);
    DAYAN_PARAMS_IF(!s.tasks || dayan_task_dim(DAYAN_TASK_Z) != 1) { return -22; }
    s.grid_rows = s.tasks == 4 && cfg->m >= 64 ? 2 : 1;
    s.grid_cols = s.tasks / s.grid_rows;
    const bool decode = cfg->m == 1 && cfg->weight_type == kDayanDataTypeMXFP4;
    s.share = cfg->share_inputs && !decode && s.tasks == 4 && dayan_task_mode() == 4;
    s.shape = DayanMxfp4Tile::make(cfg->m, cfg->n, cfg->k, cfg->weight_type == kDayanDataTypeMXFP4,
                                 cfg->operation >= 2 ? 2 : 1, cfg->operation != 0 || cfg->output_type != kDayanDataTypeF32, s.tasks, dayan_uram_size());
    switch (s.shape.m) {
        case 8: return run_tiled<8, 2048>(s);
        case 512: return run_tiled<512, 512>(s);
        case 256: return run_tiled<256, 512>(s);
        case 128: return run_tiled<128, 512>(s);
        case 64: return run_tiled<64, 512>(s);
        default:
            switch (s.shape.n) {
                case 128: return run_tiled<32, 128>(s);
                case 256: return run_tiled<32, 256>(s);
                case 512: return run_tiled<32, 512>(s);
                case 768: return run_tiled<32, 768>(s);
                case 1024: return run_tiled<32, 1024>(s);
                case 1280: return run_tiled<32, 1280>(s);
                case 1536: return run_tiled<32, 1536>(s);
                case 1792: return run_tiled<32, 1792>(s);
                case 2048: return run_tiled<32, 2048>(s);
                case 2304: return run_tiled<32, 2304>(s);
                case 2560: return run_tiled<32, 2560>(s);
                case 2816: return run_tiled<32, 2816>(s);
                case 3072: return run_tiled<32, 3072>(s);
                default: return -22;
            }
    }
}
} // namespace ggml_dayan_mxfp4_detail
#endif
