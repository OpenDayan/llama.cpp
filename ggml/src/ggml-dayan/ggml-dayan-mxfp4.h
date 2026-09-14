#pragma once

#include "ggml-dayan-matmul.h"
#include "ggml-dayan-mxfp4-layout.h"

#if defined(__DAYAN_DEVICE__)
namespace ggml_dayan_mxfp4_detail {

using namespace ggml_dayan_matmul_detail;

enum : int { PostRows = DayanMxfp4Tile::post_rows, PostStride = DayanMxfp4Tile::post_stride };

// VisualSwizzle: peak 1; keep the 320B mapping inside each 256B window.
static uint64_t row_swizzle(int stride) {
    if (stride == 320) { return MAT_BATCH_SWIZZLE(7, 1, 3); }
    int bit = 0;
    while ((stride & 1) == 0) { ++bit; stride >>= 1; }
    return bit <= 7 ? 0 : MAT_BATCH_SWIZZLE(7, bit < 10 ? bit - 7 : 3, bit < 10 ? 3 : bit - 7);
}

static void fp16_to_f32() { __tx_vec_cvt_fp32_fp16<0, 0, 0>(); }

static void quantize_tile(uint64_t input, int input_stride, uint32_t type, uint64_t raw,
                          uint64_t packed, uint64_t scales, int rows, int cols) {
    using namespace ggml_dayan_vec_detail;
    const bool bf16 = type == kDayanDataTypeBF16;
    const long instructions = bf16 ? 3 : 10;
    configure_vec_grid(1, rows);
    configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, 0, input_stride);
    configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, 0, 256);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    if (!bf16) {
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
        __tx_vec_shuffle_compress<0, 0>(0);
        __tx_vec_lsu_u2v<0, 0>(input + 256);
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
        __tx_vec_shuffle_compress<0, 0>(0);
        __tx_vec_lsu_g2s<0>((long) 0xffffffff00000000ULL);
        __tx_vec_vv_mux2<0, 3, 0, 0>();
    }
    __tx_vec_cvt_mxfp4_bf16<0, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(raw);
    start_vec_block(instructions);
    dayan_wait_vec();

    // Each subcore returns 16 payload bytes and one scale at byte 32.
    long checksum = 0;
    set_lsu_matrix<0>(checksum, rows, 64, 256, 64);
    __tx_lsu_tcsr_write<LSU_DIM0>(cols / 32 - 1);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(16);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(15);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(15);
    __tx_lsu_dma_u2u<0>(packed, raw);
    set_lsu_matrix<0>(checksum, rows, 64, 256, 4);
    __tx_lsu_tcsr_write<LSU_DIM0>(cols / 32 - 1);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(1);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(0);
    __tx_lsu_dma_u2u<0>(scales, raw + 32);
    dayan_wait_lsu();
}

static void store_quantized(uint64_t packed, uint64_t scales, uint64_t output, uint64_t output_scales,
                            int rows, int cols, uint64_t row_stride) {
    long checksum = 0;
    set_lsu_matrix<0>(checksum, rows, 64, 64, row_stride);
    store_matrix<0>(cols / 2, rows, 64, rows, packed, output);
    set_lsu_matrix<0>(checksum, 1, ((rows * 4 + 63) & ~63), rows * 4, rows * 4);
    store_matrix<0>(rows * 4, 1, ((rows * 4 + 63) & ~63), 1, scales, output_scales);
}

static void convert(const DayanMxfp4Config * cfg, const void * input, void * output, void * scales) {
    using namespace ggml_dayan_vec_detail;
    const uint64_t base = (uintptr_t) dayan_uram_base();
    const uint64_t nt = (cfg->n + 127) / 128;
    const int bytes = cfg->source_type == kDayanDataTypeF32 ? 4 : 2;
    const uint64_t values = cfg->source_type == kDayanDataTypeF16 ? base + PostRows * 256 : base;
    const uint64_t raw = values + PostRows * (cfg->source_type == kDayanDataTypeBF16 ? 256 : 512);
    const uint64_t packed = raw + PostRows * 256, scale = packed + PostRows * 64;
    if (!base || scale + PostRows * 4 + 64 - base > dayan_uram_size()) { dayan_thread_exit(-12); }
    long checksum = 0;
    for (uint64_t tile = dayan_task_id(DAYAN_TASK_X); tile < ((cfg->m + PostRows - 1) / PostRows) * nt; tile += 4) {
        const uint64_t row = tile / nt * PostRows, col = tile % nt * 128;
        const int rows = extent(cfg->m - row, PostRows), cols = extent(cfg->n - col, 128);
        set_lsu_matrix<0>(checksum, rows, 128 * bytes, cfg->a_stride, 128 * bytes);
        load_matrix<0>(cols * bytes, rows, 128 * bytes, rows, base,
                       (uintptr_t) input + row * cfg->a_stride + col * bytes);
        dayan_wait_lsu();
        uint32_t type = cfg->source_type;
        if (type == kDayanDataTypeF16) {
            issue_lowp_to_f32<fp16_to_f32>(values, base, cols, rows, 256, 512);
            type = kDayanDataTypeF32;
        }
        quantize_tile(values, type == kDayanDataTypeF32 ? 512 : 256, type, raw, packed, scale, rows, cols);
        store_quantized(packed, scale, (uintptr_t) output + row * cfg->output_stride + col / 2,
                        (uintptr_t) scales + col / 128 * cfg->output_scale_stride + row * 4,
                        rows, cols, cfg->output_stride);
        dayan_wait_lsu();
    }
}

struct State {
    const DayanMxfp4Config * cfg;
    DayanMxfp4Tile shape;
    uint64_t input[3], scales[3], output, output_scales;
    uint64_t physical, local, staging, raw, packed, packed_scales;
    uint64_t a[2], as[2], b[2], bs[2], c[2];
    int slice_n, c_stride, kt, nt;
    uint64_t tiles;
};

struct Tile { uint64_t row, col; int rows, cols, users; };

static Tile tile_at(const State & s, uint64_t out, int slice) {
    const uint64_t tile = out * 4 + dayan_task_id(DAYAN_TASK_X);
    const uint64_t row = tile / s.nt * s.shape.m, col = tile % s.nt * s.shape.n;
    const int cols = extent((int64_t) s.cfg->n - col, s.shape.n);
    return {row, col + slice * s.slice_n, extent((int64_t) s.cfg->m - row, s.shape.m),
            extent(cols - slice * s.slice_n, s.slice_n), (cols + s.slice_n - 1) / s.slice_n};
}

template <bool B>
static void load(void * arg) {
    const State & s = *(const State *) arg;
    const auto & shape = s.shape;
    long checksum = 0;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int step = 0; step < s.kt; ++step) {
            for (int slice = 0; slice < (B ? 2 : 1); ++slice) {
                const Tile tile = tile_at(s, out, slice);
                if (!tile.rows || !tile.cols) { continue; }
                const int slot = B ? slice : (out * s.kt + step) & 1;
                const int rows = B ? tile.cols : tile.rows;
                const int capacity = B ? s.slice_n : shape.m;
                const uint64_t first = B ? tile.col : tile.row;
                const auto resource = DAYAN_LOCAL_RESOURCE((B ? 2 : 0) + slot);
                matmul_check(dayan_resource_wait_free(resource));
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
                matmul_check(dayan_resource_produce_lsu(resource, (B ? 1 : tile.users) * shape.products, 0));
            }
        }
    }
    dayan_wait_lsu();
}

static void calculate(State & s) {
    const auto & shape = s.shape;
    uint64_t phases = 0;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(shape.input_stride);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(shape.input_stride);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(s.c_stride);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(row_swizzle(shape.input_stride));
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(row_swizzle(shape.input_stride));
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(row_swizzle(s.c_stride));
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_ASCALESTRIDE>(shape.a_scale_stride());
    __tx_mat_tcsr_write<MAT_BSCALESTRIDE>(s.slice_n * 4);
    const long datatype = s.cfg->weight_type == kDayanDataTypeMXFP4 ? MAT_DATATYPE_PAIR(0x38, 0x38) :
        s.cfg->weight_type == kDayanDataTypeBF16 ? BaseNPUInstBuilder::DT_BF16 : BaseNPUInstBuilder::DT_FP16;
    __tx_mat_tcsr_write<MAT_DATATYPE>(datatype);
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int step = 0; step < s.kt; ++step) {
            const int slot = (out * s.kt + step) & 1;
            for (int slice = 0; slice < 2; ++slice) {
                const Tile tile = tile_at(s, out, slice);
                if (!tile.rows || !tile.cols) { continue; }
                const auto result = DAYAN_LOCAL_RESOURCE(4 + slice);
                const auto a = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(slot));
                const auto b = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(2 + slice));
                matmul_check(dayan_resource_wait(step == 0 ? DAYAN_RESOURCE_MASK(result) : 0, a | b, phases, 0));
                __tx_mat_tcsr_write<MAT_M>(tile.rows - 1);
                __tx_mat_tcsr_write<MAT_N>(tile.cols - 1);
                __tx_mat_tcsr_write<MAT_K>(extent(s.cfg->k - step * shape.k, shape.k) - 1);
                for (int product = 0; product < shape.products; ++product) {
                    const uint64_t weight = s.b[slice] + product * s.slice_n * shape.input_stride;
                    const uint64_t c = s.c[slice] + product * shape.m * s.c_stride;
                    __tx_mat_setptr<0>(0, s.as[slot], s.bs[slice] + product * (shape.k / 128) * s.slice_n * 4);
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
    }
    dayan_wait_mat();
}

static void post(void * arg) {
    State & s = *(State *) arg;
    using namespace ggml_dayan_vec_detail;
    uint64_t phases = 0;
    long checksum = 0;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int slice = 0; slice < 2; ++slice) {
            const Tile tile = tile_at(s, out, slice);
            if (!tile.rows || !tile.cols) { continue; }
            const auto mask = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(4 + slice));
            matmul_check(dayan_resource_wait(0, mask, phases, 0));
            const uint64_t c = s.physical + s.c[slice] - s.local;
            const uint64_t destination = s.output + tile.row * s.cfg->output_stride;
            if (!s.shape.post) {
                set_lsu_matrix<0>(checksum, tile.rows, s.c_stride, s.c_stride, s.cfg->output_stride);
                __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(row_swizzle(s.c_stride));
                store_matrix<0>(tile.cols * 4, tile.rows, s.c_stride, tile.rows, c, destination + tile.col * 4);
            } else for (int row = 0; row < tile.rows; row += PostRows) {
                for (int col = 0; col < tile.cols; col += 128) {
                    const int rows = extent(tile.rows - row, PostRows), cols = extent(tile.cols - col, 128);
                    // LSU removes the MAT swizzle before VEC reads the small post tile.
                    dayan_wait_lsu();
                    set_lsu_matrix<0>(checksum, rows, 512, s.c_stride, PostStride);
                    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(row_swizzle(s.c_stride));
                    for (int product = 0; product < s.shape.products; ++product) {
                        __tx_lsu_dma_u2u<0>(s.staging + product * PostRows * PostStride,
                                           c + (product * s.shape.m + row) * s.c_stride + col * 4);
                    }
                    dayan_wait_lsu();
                    const F32Tensor gate = {s.staging, PostStride}, up = {s.staging + PostRows * PostStride, PostStride};
                    if (s.cfg->operation == 2) { issue_swiglu(gate, up, gate, rows, cols); }
                    if (s.cfg->operation == 3) { issue_geglu(gate, up, gate, rows, cols); }
                    if (s.cfg->operation == 1) {
                        MatmulEpilogueArgs args = {{s.staging, 0}, s.staging, rows, cols, PostStride / 4};
                        epilogue_silu(&args);
                    }
                    dayan_wait_vec();
                    const uint64_t dst = destination + row * s.cfg->output_stride;
                    if (s.cfg->output_type == kDayanDataTypeMXFP4) {
                        quantize_tile(s.staging, PostStride, kDayanDataTypeF32, s.raw, s.packed, s.packed_scales, rows, cols);
                        store_quantized(s.packed, s.packed_scales, dst + (tile.col + col) / 2,
                                        s.output_scales + (tile.col + col) / 128 * s.cfg->output_scale_stride + (tile.row + row) * 4,
                                        rows, cols, s.cfg->output_stride);
                    } else if (s.cfg->output_type == kDayanDataTypeF32) {
                        set_lsu_matrix<0>(checksum, rows, 512, PostStride, s.cfg->output_stride);
                        store_matrix<0>(cols * 4, rows, 512, rows, s.staging, dst + (tile.col + col) * 4);
                    } else {
                        if (s.cfg->output_type == kDayanDataTypeBF16) {
                            convert_f32_to_lowp<BaseNPUInstBuilder::DT_BF16, 0>(gate, gate, {s.packed, 256}, rows, cols);
                        } else {
                            convert_f32_to_lowp<BaseNPUInstBuilder::DT_FP16, 0>(gate, gate, {s.packed, 256}, rows, cols);
                        }
                        set_lsu_matrix<0>(checksum, rows, 256, 256, s.cfg->output_stride);
                        store_matrix<0>(cols * 2, rows, 256, rows, s.packed, dst + (tile.col + col) * 2);
                    }
                }
            }
            matmul_check(dayan_resource_consume_lsu(mask));
            phases ^= mask;
        }
    }
    dayan_wait_lsu();
}

static int run(const DayanMxfp4Config * cfg, const void * a, const void * as, const void * b, const void * bs,
               const void * up, const void * ups, void * output, void * output_scales) {
    State s = {};
    s.cfg = cfg;
    s.input[0] = (uintptr_t) a; s.input[1] = (uintptr_t) b; s.input[2] = (uintptr_t) up;
    s.scales[0] = (uintptr_t) as; s.scales[1] = (uintptr_t) bs; s.scales[2] = (uintptr_t) ups;
    s.output = (uintptr_t) output; s.output_scales = (uintptr_t) output_scales;
    s.shape = DayanMxfp4Tile::make(cfg->m, cfg->n, cfg->k, cfg->weight_type == kDayanDataTypeMXFP4,
                                 cfg->operation >= 2 ? 2 : 1, cfg->operation != 0 || cfg->output_type != kDayanDataTypeF32);
    const auto & shape = s.shape;
    s.slice_n = shape.n / 2;
    s.c_stride = s.slice_n * 4;
    s.kt = (cfg->k + shape.k - 1) / shape.k;
    s.nt = (cfg->n + shape.n - 1) / shape.n;
    s.tiles = (((cfg->m + shape.m - 1) / shape.m) * s.nt + 3) / 4;
    s.physical = (uintptr_t) dayan_uram_base();
    s.local = s.physical & (Constants::kUramWindowStride - 1);
    uint64_t address = s.local;
    for (int i = 0; i < 2; ++i) {
        s.a[i] = address; address += shape.m * shape.input_stride;
        s.b[i] = address; address += shape.products * s.slice_n * shape.input_stride;
    }
    address = s.local + shape.accumulator_offset();
    for (int i = 0; i < 2; ++i) {
        s.c[i] = address; address += shape.products * shape.m * s.c_stride;
    }
    for (int i = 0; i < 2; ++i) {
        s.as[i] = address; address += shape.scales ? shape.a_scale_stride() * (shape.k / 128) : 0;
        s.bs[i] = address; address += shape.scales ? shape.products * s.slice_n * (shape.k / 32) : 0;
    }
    s.staging = s.physical + shape.scratch_offset();
    s.raw = s.staging + shape.products * PostRows * PostStride;
    s.packed = s.raw + PostRows * 256;
    s.packed_scales = s.packed + PostRows * 256;
    if (!s.physical || shape.bytes() > dayan_uram_size() || shape.bytes() > Constants::kUramWindowStride - s.local) { return -12; }
    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = 1;
    resources.local_resource_count = 6;
    matmul_check(dayan_create_thread("mxfp4-load-a", load<false>, &s, &resources));
    matmul_check(dayan_create_thread("mxfp4-load-b", load<true>, &s, &resources));
    resources.vec_tcsr_count = 1;
    matmul_check(dayan_create_thread("mxfp4-post", post, &s, &resources));
    calculate(s);
    return dayan_thread_join_all();
}
} // namespace ggml_dayan_mxfp4_detail
#endif
