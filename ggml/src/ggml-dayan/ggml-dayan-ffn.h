#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_ffn_detail {

using ggml_dayan_matmul_detail::MatmulBuffer;
using ggml_dayan_matmul_detail::MatmulCluster;
using ggml_dayan_matmul_detail::MatmulPhases;
using ggml_dayan_matmul_detail::extent;
using ggml_dayan_matmul_detail::load_matrix;
using ggml_dayan_matmul_detail::matmul_check;
using ggml_dayan_matmul_detail::store_matrix;

enum : int {
    kSlices = 2,
    kInner = 256,
    kInputStride = kInner * sizeof(uint16_t),
};

struct State {
    const DayanMatmulConfig * config;
    uint64_t input[3], output, physical, uram, tiles, col_begin, col_end;
    uint64_t packed, load_phases;
    int lane, tile_m, slice_m, tile_n, slice_n, nt, kt;
    int accumulator_stride, output_stride;
    bool buffer_a;
    MatmulBuffer a[kSlices], b[kSlices], c[kSlices];
    MatmulPhases * phases;
};

struct Tile {
    uint64_t row, col;
    int rows, cols, half, users;
};

static inline Tile tile(const State & s, uint64_t index, int slice) {
    const uint64_t row = index / s.nt * s.tile_m;
    const uint64_t col = s.col_begin + index % s.nt * s.tile_n;
    const int rows = extent((int64_t) s.config->m - (int64_t) row, s.tile_m);
    const int cols = extent((int64_t) s.col_end - (int64_t) col, s.tile_n);
    const int row_offset = s.buffer_a ? 0 : slice * s.slice_m;
    const int col_offset = s.buffer_a ? slice * s.slice_n : 0;
    const int width = extent(cols - col_offset, s.slice_n);
    const int users = s.buffer_a ? (cols + s.slice_n - 1) / s.slice_n : (rows + s.slice_m - 1) / s.slice_m;
    return {row + row_offset, col + col_offset, extent(rows - row_offset, s.slice_m), width, (width + 63) & ~63, users};
}

template <int Cores>
static void load_a(void * argument) {
    State & s = *(State *) argument;
    MatmulCluster * cluster = (MatmulCluster *) shared;
    uint64_t phases = 0;
    if constexpr (Cores == 4) {
        for (int i = 0; i < kSlices; ++i) {
            phases |= (uint64_t) cluster->shared_phase[i] << DAYAN_SHARED_RESOURCE(i);
        }
    }
    long checksum = 0;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int step = 0; step < s.kt; ++step) {
            for (int slice = 0; slice < (s.buffer_a ? 1 : kSlices); ++slice) {
                const Tile n = tile(s, out, slice);
                if (!n.rows) {
                    continue;
                }
                const int index = s.buffer_a ? (out * s.kt + step) & 1 : slice;
                const MatmulBuffer & a = s.a[index];
                const auto local = n.cols ? DAYAN_RESOURCE_MASK(a.resource) : 0;
                const auto ready = DAYAN_SHARED_RESOURCE(index);
                const auto mask = DAYAN_RESOURCE_MASK(ready);
                const int users = s.buffer_a ? n.users : 1;
                if (Cores == 1 || s.lane == 0) {
                    matmul_check(dayan_resource_wait(local | (Cores == 4 ? mask : 0), 0, 0, 0));
                    const int inner = extent((int64_t) s.config->k - (int64_t) step * kInner, kInner);
                    set_lsu_matrix<0>(checksum, n.rows, kInputStride, s.config->a_stride, kInputStride);
                    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(ggml_dayan_matmul_detail::kMatSwizzle);
                    load_matrix<0>(inner * 2, n.rows, kInputStride, n.rows, a.uram,
                                   s.input[0] + n.row * s.config->a_stride + (uint64_t) step * kInputStride);
                    if (n.cols) {
                        matmul_check(dayan_resource_produce_lsu(a.resource, users, 0));
                    }
                    if constexpr (Cores == 4) {
                        matmul_check(dayan_resource_produce_lsu(ready, Cores - 1, 0));
                    }
                } else {
                    matmul_check(dayan_resource_wait(local, mask, phases, 0));
                    if (n.cols) {
                        set_lsu_matrix<0>(checksum, n.rows, kInputStride, kInputStride, kInputStride);
                        __tx_lsu_dma_d2u<0>(a.uram, cluster->uram_base[0] + a.uram - s.uram);
                        matmul_check(dayan_resource_produce_lsu(a.resource, users, mask));
                    } else {
                        matmul_check(dayan_resource_consume_lsu(mask));
                    }
                }
                phases ^= mask;
            }
        }
    }
    dayan_wait_lsu();
    s.load_phases = phases;
}

static void load_b(void * argument) {
    const State & s = *(const State *) argument;
    long checksum = 0;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int step = 0; step < s.kt; ++step) {
            for (int slice = 0; slice < (s.buffer_a ? kSlices : 1); ++slice) {
                const Tile n = tile(s, out, slice);
                if (!n.cols) {
                    continue;
                }
                const MatmulBuffer & b = s.b[s.buffer_a ? slice : (out * s.kt + step) & 1];
                const int inner = extent((int64_t) s.config->k - (int64_t) step * kInner, kInner);
                const uint64_t offset = n.col * s.config->bt_stride + (uint64_t) step * kInputStride;
                matmul_check(dayan_resource_wait_free(b.resource));
                set_lsu_matrix<0>(checksum, n.cols, kInputStride, s.config->bt_stride, kInputStride);
                __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(ggml_dayan_matmul_detail::kMatSwizzle);
                load_matrix<0>(inner * 2, n.cols, kInputStride, n.cols, b.uram, s.input[1] + offset);
                load_matrix<0>(inner * 2, n.cols, kInputStride, n.cols, b.uram + n.half * kInputStride, s.input[2] + offset);
                matmul_check(dayan_resource_produce_lsu(b.resource, s.buffer_a ? 1 : n.users, 0));
            }
        }
    }
    dayan_wait_lsu();
}

template <typename Format>
static void calculate(State & s) {
    uint64_t & phases = s.phases->mat;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(kInputStride);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(kInputStride);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(s.accumulator_stride);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(ggml_dayan_matmul_detail::kMatSwizzle);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(ggml_dayan_matmul_detail::kMatSwizzle);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(Format::kMatDataType);
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int step = 0; step < s.kt; ++step) {
            for (int slice = 0; slice < kSlices; ++slice) {
                const Tile n = tile(s, out, slice);
                if (!n.rows || !n.cols) {
                    continue;
                }
                const int index = (out * s.kt + step) & 1;
                const MatmulBuffer & a = s.a[s.buffer_a ? index : slice];
                const MatmulBuffer & b = s.b[s.buffer_a ? slice : index];
                const MatmulBuffer & c = s.c[slice];
                const auto inputs = DAYAN_RESOURCE_MASK(a.resource) | DAYAN_RESOURCE_MASK(b.resource);
                matmul_check(dayan_resource_wait(step == 0 ? DAYAN_RESOURCE_MASK(c.resource) : 0, inputs, phases, 0));
                __tx_mat_tcsr_write<MAT_M>(n.rows - 1);
                __tx_mat_tcsr_write<MAT_N>(n.half + n.cols - 1);
                __tx_mat_tcsr_write<MAT_K>(extent((int64_t) s.config->k - (int64_t) step * kInner, kInner) - 1);
                if (step == 0) {
                    __tx_mat_mm<0>(c.uram, a.uram, b.uram);
                } else {
                    __tx_mat_mma<0>(c.uram, a.uram, b.uram);
                }
                if (step + 1 == s.kt) {
                    matmul_check(dayan_resource_produce_mat(c.resource, 1, inputs));
                } else {
                    matmul_check(dayan_resource_consume_mat(inputs));
                }
                // The buffered input stays ready until all active slices consume it.
                phases ^= DAYAN_RESOURCE_MASK(s.buffer_a ? b.resource : a.resource);
                if (slice + 1 == n.users) {
                    phases ^= DAYAN_RESOURCE_MASK(s.buffer_a ? a.resource : b.resource);
                }
            }
        }
    }
    dayan_wait_mat();
}

template <typename Format, bool GeGLU>
static void post(void * argument) {
    State & s = *(State *) argument;
    uint64_t & phases = s.phases->post;
    long checksum = 0;
    using namespace ggml_dayan_vec_detail;
    for (uint64_t out = 0; out < s.tiles; ++out) {
        for (int slice = 0; slice < kSlices; ++slice) {
            const Tile n = tile(s, out, slice);
            if (!n.rows || !n.cols) {
                continue;
            }
            const MatmulBuffer & c = s.c[slice];
            const auto mask = DAYAN_RESOURCE_MASK(c.resource);
            matmul_check(dayan_resource_wait(0, mask, phases, 0));
            // LSU must finish reading the shared packed buffer before VEC reuses it.
            dayan_wait_lsu();
            const F32Tensor gate = {s.physical + c.uram - s.uram, s.accumulator_stride};
            const F32Tensor up = {gate.address + n.half * sizeof(float), s.accumulator_stride};
            const F32Tensor output = {s.packed, s.output_stride};
            configure_activation(gate, up, output, n.rows, n.cols);
            __tx_vec_tcsr_write<kVecV3Base + 3>(kVecLanes * sizeof(uint16_t));
            if constexpr (GeGLU) {
                issue_geglu_configured<Format::kMatDataType, true>(gate, up, output);
            } else {
                issue_swiglu_configured<Format::kMatDataType, true>(gate, up, output);
            }
            dayan_wait_vec();
            set_lsu_matrix<0>(checksum, n.rows, s.output_stride, s.output_stride, s.config->output_stride);
            store_matrix<0>(n.cols * 2, n.rows, s.output_stride, n.rows, output.address,
                            s.output + n.row * s.config->output_stride + n.col * 2);
            matmul_check(dayan_resource_consume_lsu(mask));
            phases ^= mask;
        }
    }
    dayan_wait_lsu();
}

template <typename Format, int Cores, bool GeGLU, bool Inference = false>
static int run(const DayanMatmulConfig * config, const uint16_t * a, const uint16_t * gate, const uint16_t * up,
               uint16_t * output, MatmulPhases * phases = nullptr) {
    static_assert(Cores == 1 || Cores == 4);
    const int lane = dayan_task_id(DAYAN_TASK_X);
    if (!ggml_dayan_matmul_detail::matmul_config_valid<sizeof(uint16_t)>(config) || !a || !gate || !up || !output ||
        dayan_task_mode() != Cores || dayan_task_dim(DAYAN_TASK_X) != Cores || lane >= Cores ||
        dayan_task_dim(DAYAN_TASK_Y) != 1 || dayan_task_dim(DAYAN_TASK_Z) != 1 || (Cores == 4 && !shared) ||
        (Inference && config->m > 512)) {
        return -22;
    }
    MatmulPhases local_phases;
    MatmulCluster * cluster = (MatmulCluster *) shared;
    State s = {};
    s.config = config;
    s.lane = lane;
    s.buffer_a = config->m <= 128;
    s.slice_m = s.buffer_a ? 128 : 256;
    s.slice_n = s.buffer_a ? 192 : 128;
    s.tile_m = s.buffer_a ? s.slice_m : kSlices * s.slice_m;
    s.tile_n = s.buffer_a ? kSlices * s.slice_n : s.slice_n;
    s.accumulator_stride = (2 * s.slice_n + 32) * sizeof(float);
    s.output_stride = s.slice_n * sizeof(uint16_t);
    s.col_begin = config->n * lane / Cores;
    s.col_end = config->n * (lane + 1) / Cores;
    // All cores must participate in the same A broadcasts, including empty N shards.
    s.nt = (int) (((config->n + Cores - 1) / Cores + s.tile_n - 1) / s.tile_n);
    s.kt = (int) ((config->k + kInner - 1) / kInner);
    s.tiles = (config->m + s.tile_m - 1) / s.tile_m * s.nt;
    if (s.tiles > UINT64_MAX / s.kt) {
        return -22;
    }
    s.phases = Cores == 4 ? &cluster->phases[lane] : phases ? phases : &local_phases;
    s.physical = (uintptr_t) dayan_uram_base();
    s.uram = s.physical & (ggml_dayan_matmul_detail::Constants::kUramWindowStride - 1);
    uint64_t address = s.uram;
    for (int i = 0; i < kSlices; ++i) {
        s.a[i] = {address, DAYAN_LOCAL_RESOURCE(i)};
        address += (uint64_t) s.slice_m * kInputStride;
    }
    for (int i = 0; i < kSlices; ++i) {
        s.b[i] = {address, DAYAN_LOCAL_RESOURCE(4 + i)};
        address += (uint64_t) 2 * s.slice_n * kInputStride;
    }
    for (int i = 0; i < kSlices; ++i) {
        s.c[i] = {address, DAYAN_LOCAL_RESOURCE(8 + i)};
        address += (uint64_t) s.slice_m * s.accumulator_stride;
    }
    s.packed = s.physical + address - s.uram;
    address += (uint64_t) s.slice_m * s.output_stride;
    if (!s.physical || address - s.uram > dayan_uram_size() || address > ggml_dayan_matmul_detail::Constants::kUramWindowStride) {
        return -12;
    }
    if constexpr (Cores == 4) {
        cluster->uram_base[lane] = s.physical;
        matmul_check(dayan_cluster_barrier(4, &cluster->barrier));
    }
    const uint64_t ne2 = config->batch_ne2 ? config->batch_ne2 : 1;
    const uint64_t ne3 = config->batch_ne3 ? config->batch_ne3 : 1;
    const uint64_t ratio2 = config->batch_ne2 ? ne2 / config->weight_ne2 : 1;
    const uint64_t ratio3 = config->batch_ne3 ? ne3 / config->weight_ne3 : 1;
    for (uint64_t i3 = 0; i3 < ne3; ++i3) {
        for (uint64_t i2 = 0; i2 < ne2; ++i2) {
            const uint64_t weight_offset = i3 / ratio3 * config->bt_batch_stride3 + i2 / ratio2 * config->bt_batch_stride2;
            s.input[0] = (uintptr_t) a + i3 * config->a_batch_stride3 + i2 * config->a_batch_stride2;
            s.input[1] = (uintptr_t) gate + weight_offset;
            s.input[2] = (uintptr_t) up + weight_offset;
            s.output = (uintptr_t) output + i3 * config->output_batch_stride3 + i2 * config->output_batch_stride2;
            dayan_thread_resources_t resources = {};
            resources.lsu_tcsr_count = 1;
            resources.local_resource_count = 12;
            resources.shared_resource_count = Cores == 4 ? kSlices : 0;
            matmul_check(dayan_create_thread("ffn-load-a", load_a<Cores>, &s, &resources));
            matmul_check(dayan_create_thread("ffn-load-b", load_b, &s, &resources));
            resources.vec_tcsr_count = 1;
            matmul_check(dayan_create_thread("ffn-post", post<Format, GeGLU>, &s, &resources));
            calculate<Format>(s);
            matmul_check(dayan_thread_join_all());
            if constexpr (Cores == 4) {
                matmul_check(dayan_cluster_barrier(4, &cluster->barrier));
                if (lane == 0) {
                    for (int i = 0; i < kSlices; ++i) {
                        cluster->shared_phase[i] = (s.load_phases >> DAYAN_SHARED_RESOURCE(i)) & 1;
                    }
                }
                matmul_check(dayan_cluster_barrier(4, &cluster->barrier));
            }
        }
    }
    return 0;
}

} // namespace ggml_dayan_ffn_detail

#endif
