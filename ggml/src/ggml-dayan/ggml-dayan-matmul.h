#pragma once

#include "ggml-dayan-activation.h"

#if defined(__DAYAN_DEVICE__)

#    include <dayan_intrinsics.h>
#    include <dayan_sys.h>

namespace ggml_dayan_matmul_detail {

using ggml_dayan_vec_detail::begin_vec_block;
using ggml_dayan_vec_detail::configure_vec_grid;
using ggml_dayan_vec_detail::configure_vec_iterator;
using ggml_dayan_vec_detail::start_vec_block;

static constexpr long kVecSync   = ggml_dayan_vec_detail::kVecSync;
static constexpr long kVecV0Base = ggml_dayan_vec_detail::kVecV0Base;
static constexpr long kVecV1Base = ggml_dayan_vec_detail::kVecV1Base;
static constexpr long kVecV2Base = ggml_dayan_vec_detail::kVecV2Base;
static constexpr long kVecV3Base = ggml_dayan_vec_detail::kVecV3Base;

struct Constants {
    enum : int {
        kLsuShared           = 0,
        kTileM               = 128,
        kTileN               = 384,
        kTileK               = 256,
        kUramAlignment       = 32,
        kUramWindowStride    = 2 * 1024 * 1024,
        kVecLanes            = 64,
        kVecBytes            = kVecLanes * sizeof(float),
        kConstantStride      = 32,
        kConstantCount       = 7,
    };
};

struct Bf16Format {
    enum : long {
        kMatDataType = BaseNPUInstBuilder::DT_BF16,
    };
};

struct Fp16Format {
    enum : long {
        kMatDataType = BaseNPUInstBuilder::DT_FP16,
    };
};

static constexpr uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static inline int min_int(int left, int right) {
    return left < right ? left : right;
}

static inline uint32_t matmul_group_id() {
    return dayan_task_id(dayan_task_mode() == 4 ? DAYAN_TASK_Y : DAYAN_TASK_X);
}

static inline uint32_t matmul_group_count() {
    return dayan_task_dim(dayan_task_mode() == 4 ? DAYAN_TASK_Y : DAYAN_TASK_X);
}

struct MatmulInputArgs {
    const void * context;
    uint64_t     output;
    uint64_t     scratch;
    int          row_base;
    int          k_base;
    int          rows;
    int          cols;
};

using MatmulInputLoader = int (*)(const MatmulInputArgs * args);

struct MatmulEpilogueArgs {
    uint64_t accumulator[2];
    uint64_t output;
    int      rows;
    int      cols;
    int      row_stride;
};

using MatmulEpilogue = uint64_t (*)(const MatmulEpilogueArgs * args);

template <long TcsrId>
static inline void load_matrix(uint64_t major_left,
                               uint64_t minor_left,
                               uint64_t major_tile,
                               uint64_t minor_tile,
                               uint64_t uram_base,
                               uint64_t dram_base) {
    if (minor_left == 0 || major_left == 0) {
        return;
    }

    const uint64_t real_minor = minor_left < minor_tile ? minor_left : minor_tile;
    __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM1>(real_minor - 1U);

    if (major_left >= major_tile) {
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(major_tile / 64U - 1U);
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
        __tx_lsu_dma_d2u<TcsrId>((long) uram_base, (long) dram_base);
        return;
    }

    while (major_left) {
        uint64_t       handled_bytes   = major_left & ~0x3fULL;
        const uint64_t remaining_bytes = major_left & 0x3fULL;

        if (handled_bytes == 0 && remaining_bytes > 0) {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(0);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(remaining_bytes - 1U);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
            handled_bytes = remaining_bytes;
        } else {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(handled_bytes / 64U - 1U);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
        }
        __tx_lsu_dma_d2u<TcsrId>((long) uram_base, (long) dram_base);
        uram_base += handled_bytes;
        dram_base += handled_bytes;
        major_left -= handled_bytes;
    }
}

template <long TcsrId>
static inline void store_matrix(uint64_t major_left,
                                uint64_t minor_left,
                                uint64_t major_tile,
                                uint64_t minor_tile,
                                uint64_t uram_base,
                                uint64_t dram_base) {
    if (minor_left == 0 || major_left == 0) {
        return;
    }

    const uint64_t real_minor = minor_left < minor_tile ? minor_left : minor_tile;
    __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM1>(real_minor - 1U);

    if (major_left >= major_tile) {
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(major_tile / 64U - 1U);
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
        __tx_lsu_dma_u2d<TcsrId>((long) dram_base, (long) uram_base);
        return;
    }

    while (major_left) {
        uint64_t       handled_bytes   = major_left & ~0x3fULL;
        const uint64_t remaining_bytes = major_left & 0x3fULL;

        if (handled_bytes == 0 && remaining_bytes > 0) {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(0);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(remaining_bytes - 1U);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(remaining_bytes - 1U);
            handled_bytes = remaining_bytes;
        } else {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(handled_bytes / 64U - 1U);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
        }
        __tx_lsu_dma_u2d<TcsrId>((long) dram_base, (long) uram_base);
        uram_base += handled_bytes;
        dram_base += handled_bytes;
        major_left -= handled_bytes;
    }
}

template <long MatDataType, long LsuGroup> static inline void quantize(const MatmulEpilogueArgs * args) {
    const ggml_dayan_vec_detail::F32Tensor accumulator = {
        args->accumulator[0], args->row_stride * (int) sizeof(float)
    };
    ggml_dayan_vec_detail::convert_f32_to_lowp<MatDataType, LsuGroup>(
        accumulator, accumulator,
        { args->output, args->row_stride * (int) sizeof(uint16_t) }, args->rows, args->cols);
}

static uint64_t epilogue_identity(const MatmulEpilogueArgs * args) {
    return args->accumulator[0];
}

template <typename Format>
static uint64_t epilogue_lowp(const MatmulEpilogueArgs * args) {
    quantize<Format::kMatDataType, 0>(args);
    return args->output;
}

template <typename Format, bool GeGLU>
static uint64_t epilogue_glu(const MatmulEpilogueArgs * args) {
    const long stride = args->row_stride * (int) sizeof(float);
    const ggml_dayan_vec_detail::F32Tensor gate = {args->accumulator[0], stride};
    const ggml_dayan_vec_detail::F32Tensor up = {args->accumulator[1], stride};
    if constexpr (GeGLU) {
        ggml_dayan_vec_detail::issue_geglu<Format::kMatDataType>(gate, up, gate, args->rows, args->cols);
    } else {
        ggml_dayan_vec_detail::issue_swiglu<Format::kMatDataType>(gate, up, gate, args->rows, args->cols);
    }
    dayan_wait_vec();
    ggml_dayan_vec_detail::copy_packed_lowp<0>(gate, {args->output, args->row_stride * (int) sizeof(uint16_t)},
                                             args->rows, (args->cols + Constants::kVecLanes - 1) / Constants::kVecLanes);
    return args->output;
}

static uint64_t epilogue_silu(const MatmulEpilogueArgs * args) {
    const ggml_dayan_vec_detail::F32Tensor tile = {args->accumulator[0], args->row_stride * (int) sizeof(float)};
    constexpr long instructions = 9;
    ggml_dayan_vec_detail::configure_activation(tile, tile, tile, args->rows, args->cols);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(tile.address);
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) ggml_dayan_vec_detail::f32_bits(-1.4426950408889634f));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_vv_fp32_mul<0, 0, 4, 0>();
    __tx_vec_lsu_v2u<0, 3>(tile.address);
    start_vec_block(instructions);
    dayan_wait_vec();
    return tile.address;
}

struct MatmulPhases {
    uint64_t mat = 0;
    uint64_t post = 0;
};

struct MatmulCluster {
    dayan_cluster_barrier_t barrier;
    int32_t selected_status;
    int32_t selected_expert;
    uint64_t uram_base[4];
    MatmulPhases phases[4];
    uint32_t shared_phase[12];
};

static_assert(sizeof(MatmulCluster) <= MATMUL_DAYAN_4C_ATOMIC_BYTES, "MatMul shared state exceeds its allocation");

static constexpr uint64_t kMatSwizzle = MAT_BATCH_SWIZZLE(7, 2, 3);
static constexpr int kMatmulSlices = 4;

struct MatmulBuffer {
    uint64_t uram;
    dayan_resource_t resource;
};

struct MatmulState {
    const DayanMatmulConfig * config;
    uint64_t input[3], output, physical, uram, generations;
    uint64_t accumulator_bytes, output_bytes, output_start, scratch;
    uint64_t a_swizzle, c_swizzle, load_phases[2];
    int cores, lane, tile_m, tile_n, nt, kt, slice_m, slice_n, c_columns;
    bool buffer_a, split_n;
    MatmulBuffer a[4], b[4], c[4];
    MatmulPhases * phases;
    const void * input_context;
};

struct MatmulLoadArgs {
    MatmulState * state;
    bool b;
};

struct MatmulNode {
    uint64_t row, col;
    int rows, cols, inner, step, a_index, b_index, users;
};

static inline int extent(int64_t remaining, int size) {
    return remaining <= 0 ? 0 : (remaining < size ? (int) remaining : size);
}

static inline MatmulNode matmul_node(const MatmulState & s, uint64_t generation, int product, int products, int slice, int lane) {
    const uint64_t out = generation / s.kt;
    const int step = (int) (generation % s.kt);
    const int grid = s.cores == 4 && !s.split_n ? 2 : 1;
    const uint64_t begin = s.split_n ? 0 : s.config->m * (lane >> 1) / grid;
    const uint64_t end = s.split_n ? s.config->m : s.config->m * ((lane >> 1) + 1) / grid;
    const uint64_t row = begin + out / s.nt * s.tile_m;
    const uint64_t col_base = out % s.nt * grid * s.tile_n;
    // Keep full tiles large and divide the final column group across core columns.
    const int tile_n = s.split_n ? s.tile_n : (extent((int64_t) s.config->n - (int64_t) col_base, grid * s.tile_n) + grid - 1) / grid;
    const uint64_t col_begin = s.split_n ? s.config->n * lane / s.cores : (lane & 1) * tile_n;
    const uint64_t col_end = s.split_n ? s.config->n * (lane + 1) / s.cores : s.config->n;
    const uint64_t col = col_begin + col_base;
    const int rows = extent((int64_t) end - (int64_t) row, s.tile_m);
    const int cols = extent((int64_t) col_end - (int64_t) col, tile_n);
    const int row_offset = s.buffer_a ? 0 : slice * s.slice_m;
    const int col_offset = s.buffer_a ? slice * s.slice_n : 0;
    const int users = s.buffer_a ? (cols + s.slice_n - 1) / s.slice_n : (rows + s.slice_m - 1) / s.slice_m;
    int b_index = s.buffer_a ? slice : (int) ((generation * products + product) & 1);
    if (s.split_n) {
        // Use all B buffers to prefetch K steps when only a few N slices are active.
        b_index = (int) (((generation * products + product) * users + slice) % kMatmulSlices);
    }
    return {row + row_offset, col + col_offset, extent(rows - row_offset, s.slice_m), extent(cols - col_offset, s.slice_n),
            extent((int64_t) s.config->k - (int64_t) step * Constants::kTileK, Constants::kTileK), step,
            s.buffer_a ? (int) (generation & 1) : slice,
            b_index, users};
}

static inline void matmul_check(int status) {
    DAYAN_PARAMS_IF(status < 0) {
        DAYAN_PARAMS_ERROR();
    }
}

template <int Cores, int Products, MatmulInputLoader InputLoader>
static void matmul_load(void * argument) {
    const MatmulLoadArgs & args = *(MatmulLoadArgs *) argument;
    const MatmulState & s = *args.state;
    const bool b = args.b;
    const bool buffered = b ? !s.buffer_a : s.buffer_a;
    const bool share = Cores == 4 && (!s.split_n || !b);
    const int peer = s.split_n ? 0 : s.lane ^ (b ? 2 : 1);
    const int pair = s.split_n ? 0 : b ? s.lane & 1 : s.lane >> 1;
    uint64_t phases = 0;
    if constexpr (Cores == 4) {
        const MatmulCluster & cluster = *(MatmulCluster *) shared;
        for (int i = 0; i < 12; ++i) {
            phases |= (uint64_t) cluster.shared_phase[i] << DAYAN_SHARED_RESOURCE(i);
        }
    }
    long checksum = 0;
    for (uint64_t gen = 0; gen < s.generations; ++gen) {
        for (int product = 0; product < (b ? Products : 1); ++product) {
            for (int slice = 0; slice < (buffered ? 1 : kMatmulSlices); ++slice) {
                const MatmulNode n = matmul_node(s, gen, product, Products, slice, s.lane);
                const MatmulNode other = matmul_node(s, gen, product, Products, slice, Cores == 4 ? peer : s.lane);
                const bool active = n.rows && n.cols;
                if (s.split_n && !b ? !n.rows : !active && !(share && other.rows && other.cols)) {
                    continue;
                }
                const MatmulBuffer & dst = b ? s.b[n.b_index] : s.a[n.a_index];
                const int index = buffered ? 2 * kMatmulSlices + pair * 2 + (b ? n.b_index : n.a_index) : pair * kMatmulSlices + slice;
                const auto ready = DAYAN_SHARED_RESOURCE(index);
                const auto ready_mask = DAYAN_RESOURCE_MASK(ready);
                const int rows = b ? n.cols : n.rows;
                const uint32_t users = (buffered ? n.users : 1) * (b ? 1 : Products);
                int role = DAYAN_RESOURCE_ROLE_PRODUCER;
                if (share && s.split_n) {
                    // Core 0 broadcasts the small A tile; each peer retires one copy credit.
                    const uint64_t local = active ? DAYAN_RESOURCE_MASK(dst.resource) : 0;
                    if (s.lane == 0) {
                        matmul_check(dayan_resource_wait(local | ready_mask, 0, 0, 0));
                    } else {
                        matmul_check(dayan_resource_wait(local, ready_mask, phases, 0));
                        role = DAYAN_RESOURCE_ROLE_CONSUMER;
                    }
                } else if (share) {
                    // Reuse needs local MAT completion and completion of the peer copy.
                    role = dayan_resource_wait(active ? DAYAN_RESOURCE_MASK(dst.resource) | ready_mask : 0, ready_mask, phases, 0);
                } else {
                    matmul_check(dayan_resource_wait_free(dst.resource));
                }
                matmul_check(role);
                if (!active && !(s.split_n && !b && s.lane == 0)) {
                    matmul_check(dayan_resource_consume_lsu(ready_mask));
                } else if (role == DAYAN_RESOURCE_ROLE_PRODUCER) {
                    if (!b && InputLoader != nullptr) {
                        const MatmulInputArgs input = {s.input_context, s.physical + dst.uram - s.uram, s.scratch,
                                                       (int) n.row, n.step * Constants::kTileK, n.rows, n.inner};
                        matmul_check(InputLoader(&input));
                    } else {
                        const uint64_t stride = b ? s.config->bt_stride : s.config->a_stride;
                        const uint64_t src = s.input[b ? 1 + product : 0] + (b ? n.col : n.row) * stride + (uint64_t) n.step * Constants::kTileK * 2;
                        set_lsu_matrix<0>(checksum, rows, Constants::kTileK * 2, stride, Constants::kTileK * 2);
                        __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(b ? kMatSwizzle : s.a_swizzle);
                        load_matrix<0>(n.inner * 2, rows, Constants::kTileK * 2, rows, dst.uram, src);
                    }
                    if (active) {
                        matmul_check(dayan_resource_produce_lsu(dst.resource, users, 0));
                    }
                    if (share) {
                        matmul_check(dayan_resource_produce_lsu(ready, s.split_n ? Cores - 1 : 1, 0));
                    }
                } else if (role == DAYAN_RESOURCE_ROLE_CONSUMER) {
                    const MatmulCluster & cluster = *(MatmulCluster *) shared;
                    const uint64_t src = cluster.uram_base[peer] + dst.uram - s.uram;
                    set_lsu_matrix<0>(checksum, rows, Constants::kTileK * 2, Constants::kTileK * 2, Constants::kTileK * 2);
                    __tx_lsu_dma_d2u<0>((long) dst.uram, (long) src);
                    matmul_check(dayan_resource_produce_lsu(dst.resource, users, ready_mask));
                } else {
                    DAYAN_PARAMS_ERROR();
                }
                phases ^= ready_mask;
            }
        }
    }
    dayan_wait_lsu();
    args.state->load_phases[b] = phases;
}

template <typename Format, int Products>
static void matmul_calc(MatmulState & s) {
    uint64_t & phases = s.phases->mat;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(Constants::kTileK * 2);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(Constants::kTileK * 2);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(s.c_columns * 4);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(s.a_swizzle);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(kMatSwizzle);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(s.c_swizzle);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(Format::kMatDataType);
    for (uint64_t gen = 0; gen < s.generations; ++gen) {
        for (int product = 0; product < Products; ++product) {
            for (int slice = 0; slice < kMatmulSlices; ++slice) {
                const MatmulNode n = matmul_node(s, gen, product, Products, slice, s.lane);
                if (!n.rows || !n.cols) {
                    continue;
                }
                const MatmulBuffer & a = s.a[n.a_index], & b = s.b[n.b_index], & c = s.c[slice];
                const auto inputs = DAYAN_RESOURCE_MASK(a.resource) | DAYAN_RESOURCE_MASK(b.resource);
                matmul_check(dayan_resource_wait(n.step == 0 && product == 0 ? DAYAN_RESOURCE_MASK(c.resource) : 0, inputs, phases, 0));
                __tx_mat_tcsr_write<MAT_M>(n.rows - 1);
                __tx_mat_tcsr_write<MAT_N>(n.cols - 1);
                __tx_mat_tcsr_write<MAT_K>(n.inner - 1);
                const uint64_t accumulator = c.uram + product * s.accumulator_bytes;
                if (n.step == 0) {
                    __tx_mat_mm<0>((long) accumulator, (long) a.uram, (long) b.uram);
                } else {
                    __tx_mat_mma<0>((long) accumulator, (long) a.uram, (long) b.uram);
                }
                if (n.step + 1 == s.kt && product + 1 == Products) {
                    matmul_check(dayan_resource_produce_mat(c.resource, 1, inputs));
                } else {
                    matmul_check(dayan_resource_consume_mat(inputs));
                }
                if (s.buffer_a || slice + 1 == n.users) {
                    phases ^= DAYAN_RESOURCE_MASK(b.resource);
                }
                // A remains READY through all gate/up products.
                if (product + 1 == Products && (!s.buffer_a || slice + 1 == n.users)) {
                    phases ^= DAYAN_RESOURCE_MASK(a.resource);
                }
            }
        }
    }
    dayan_wait_mat();
}

template <int Products, int OutputBytes, bool HasEpilogue, MatmulEpilogue Epilogue>
static void matmul_post(void * argument) {
    MatmulState & s = *(MatmulState *) argument;
    uint64_t & phases = s.phases->post;
    long checksum = 0;
    for (uint64_t gen = s.kt - 1; gen < s.generations; gen += s.kt) {
        for (int slice = 0; slice < kMatmulSlices; ++slice) {
            const MatmulNode n = matmul_node(s, gen, 0, Products, slice, s.lane);
            if (!n.rows || !n.cols) {
                continue;
            }
            const MatmulBuffer & c = s.c[slice];
            const auto mask = DAYAN_RESOURCE_MASK(c.resource);
            matmul_check(dayan_resource_wait(0, mask, phases, 0));
            uint64_t result = c.uram;
            if constexpr (HasEpilogue) {
                const uint64_t accumulator = s.physical + c.uram - s.uram;
                const MatmulEpilogueArgs args = {{accumulator, accumulator + s.accumulator_bytes},
                                                s.physical + s.output_start + slice * s.output_bytes,
                                                n.rows, n.cols, s.c_columns};
                result = Epilogue(&args);
            }
            set_lsu_matrix<0>(checksum, n.rows, s.c_columns * OutputBytes, s.c_columns * OutputBytes, s.config->output_stride);
            __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(s.c_swizzle);
            store_matrix<0>(n.cols * OutputBytes, n.rows, s.c_columns * OutputBytes, n.rows, result,
                            s.output + n.row * s.config->output_stride + n.col * OutputBytes);
            matmul_check(dayan_resource_consume_lsu(mask));
            phases ^= mask;
        }
    }
    dayan_wait_lsu();
}

static inline bool batch_stride_valid(uint64_t rows, uint64_t row_stride, uint64_t ne2, uint64_t ne3, uint64_t stride2, uint64_t stride3) {
    return rows <= UINT64_MAX / row_stride && stride2 >= rows * row_stride && ne2 <= UINT64_MAX / stride2 &&
           stride3 >= ne2 * stride2 && ne3 <= UINT64_MAX / stride3;
}

template <int OutputBytes>
static inline bool matmul_config_valid(const DayanMatmulConfig * c) {
    const uint64_t limit = 0x7ffff000ULL;
    if (!c || !c->m || !c->n || !c->k || c->m > limit || c->n > limit || c->k > limit ||
        c->a_stride < c->k * 2 || c->bt_stride < c->k * 2 || c->output_stride < c->n * OutputBytes ||
        (c->a_stride & 1) || (c->bt_stride & 1) || c->output_stride % OutputBytes ||
        c->a_stride > INT64_MAX / c->m || c->bt_stride > INT64_MAX / c->n || c->output_stride > INT64_MAX / c->m) {
        return false;
    }
    if (!(c->batch_ne2 | c->batch_ne3 | c->weight_ne2 | c->weight_ne3 | c->a_batch_stride2 | c->a_batch_stride3 |
          c->bt_batch_stride2 | c->bt_batch_stride3 | c->output_batch_stride2 | c->output_batch_stride3)) {
        return true;
    }
    return c->batch_ne2 && c->batch_ne3 && c->weight_ne2 && c->weight_ne3 &&
           c->batch_ne2 % c->weight_ne2 == 0 && c->batch_ne3 % c->weight_ne3 == 0 &&
           batch_stride_valid(c->m, c->a_stride, c->batch_ne2, c->batch_ne3, c->a_batch_stride2, c->a_batch_stride3) &&
           batch_stride_valid(c->n, c->bt_stride, c->weight_ne2, c->weight_ne3, c->bt_batch_stride2, c->bt_batch_stride3) &&
           batch_stride_valid(c->m, c->output_stride, c->batch_ne2, c->batch_ne3, c->output_batch_stride2, c->output_batch_stride3);
}

template <typename Format, int Cores, int Products, int OutputBytes, bool HasEpilogue, MatmulEpilogue Epilogue,
          MatmulInputLoader InputLoader = nullptr, uint64_t InputScratchBytes = 0, bool Inference = false>
static int run_matmul(const DayanMatmulConfig * config, const uint16_t * a, const uint16_t * bt0, const uint16_t * bt1,
                      void * output, const void * input_context = nullptr, MatmulPhases * phases = nullptr, uint64_t alignment = 64) {
    static_assert(Cores == 1 || Cores == 4);
    static_assert(Products == 1 || Products == 2);
    const int lane = Cores == 4 ? dayan_task_id(DAYAN_TASK_X) : 0;
    const uint32_t groups = matmul_group_count();
    const uint32_t group = matmul_group_id();
    if (!matmul_config_valid<OutputBytes>(config) || !a || !bt0 || !output || (Products == 2 && !bt1) ||
        dayan_task_mode() != Cores || (Cores == 4 && dayan_task_dim(DAYAN_TASK_X) != 4) || lane >= Cores ||
        !groups || group >= groups || (Cores == 1 && dayan_task_dim(DAYAN_TASK_Y) != 1) ||
        dayan_task_dim(DAYAN_TASK_Z) != 1 || (Cores == 4 && !shared) ||
        (InputLoader != nullptr && (Cores != 1 || config->m >= config->n || config->batch_ne2))) {
        return -22;
    }
    if (Inference && config->m > 512) {
        return -22;
    }
    const uint64_t begin = dayan_partition_begin(config->n, group, groups, alignment);
    const uint64_t end = dayan_partition_begin(config->n, group + 1, groups, alignment);
    if (begin == end) { return 0; }
    DayanMatmulConfig shard = *config;
    shard.n = end - begin;
    bt0 = (const uint16_t *) ((const char *) bt0 + begin * config->bt_stride);
    if (bt1) { bt1 = (const uint16_t *) ((const char *) bt1 + begin * config->bt_stride); }
    output = (char *) output + begin * OutputBytes;
    config = &shard;
    const uint64_t ne2 = config->batch_ne2 ? config->batch_ne2 : 1;
    const uint64_t ne3 = config->batch_ne3 ? config->batch_ne3 : 1;
    const uint64_t ratio2 = config->batch_ne2 ? ne2 / config->weight_ne2 : 1;
    const uint64_t ratio3 = config->batch_ne3 ? ne3 / config->weight_ne3 : 1;
    const bool packed2 = config->batch_ne2 && config->a_batch_stride2 == config->m * config->a_stride &&
                         config->output_batch_stride2 == config->m * config->output_stride;
    const bool packed3 = packed2 && config->weight_ne2 == 1 && config->a_batch_stride3 == ne2 * config->a_batch_stride2 &&
                         config->output_batch_stride3 == ne2 * config->output_batch_stride2;
    uint64_t group2 = packed2 ? ratio2 : 1;
    uint64_t group3 = packed3 ? ratio3 : 1;
    if (config->m > 0x7ffff000ULL / group2) {
        group2 = 1;
        group3 = 1;
    }
    if (config->m * group2 > 0x7ffff000ULL / group3) {
        group3 = 1;
    }
    DayanMatmulConfig matrix = *config;
    if constexpr (Inference) {
        if (config->m * group2 * group3 > 512) {
            group2 = 1;
            group3 = 1;
        }
    }
    matrix.m *= group2 * group3;
    MatmulPhases local_phases;
    MatmulCluster * cluster = (MatmulCluster *) shared;
    MatmulState s = {};
    s.config = &matrix;
    s.cores = Cores;
    s.lane = lane;
    s.tile_m = Products == 2 ? 96 : Cores == 4 && !HasEpilogue ? 512 : Constants::kTileM;
    s.tile_n = Constants::kTileN;
    s.split_n = Inference && Cores == 4;
    s.buffer_a = Inference || InputLoader != nullptr || matrix.m < matrix.n;
    if (Inference && matrix.m < 32) {
        s.tile_m = (int) matrix.m;
        s.tile_n = Cores == 4 ? 1536 : 768;
    }
    s.slice_m = s.buffer_a ? s.tile_m : s.tile_m / kMatmulSlices;
    s.slice_n = s.buffer_a ? s.tile_n / kMatmulSlices : s.tile_n;
    // Pad complete VEC chunks and rotate the linear row start across Banks.
    s.c_columns = HasEpilogue || s.tile_n != Constants::kTileN ? (int) align_up(s.slice_n, Constants::kVecLanes) + 32 : s.slice_n;
    s.a_swizzle = InputLoader == nullptr ? kMatSwizzle : 0;
    s.c_swizzle = HasEpilogue || s.buffer_a ? 0 : kMatSwizzle;
    s.physical = (uintptr_t) dayan_uram_base();
    s.uram = s.physical & (Constants::kUramWindowStride - 1);
    s.input_context = input_context;
    s.phases = Cores == 4 ? &cluster->phases[lane] : phases ? phases : &local_phases;
    const int counts[] = {s.buffer_a ? 2 : kMatmulSlices, s.buffer_a ? kMatmulSlices : 2};
    const uint64_t bytes[] = {(uint64_t) s.slice_m * Constants::kTileK * 2, (uint64_t) s.slice_n * Constants::kTileK * 2};
    uint64_t address = s.uram;
    for (int operand = 0; operand < 2; ++operand) {
        MatmulBuffer * buffers = operand ? s.b : s.a;
        for (int i = 0; i < counts[operand]; ++i) {
            buffers[i] = {address, DAYAN_LOCAL_RESOURCE(operand * kMatmulSlices + i)};
            address += bytes[operand];
        }
    }
    s.accumulator_bytes = (uint64_t) s.slice_m * s.c_columns * sizeof(float);
    for (int i = 0; i < kMatmulSlices; ++i) {
        s.c[i] = {address, DAYAN_LOCAL_RESOURCE(2 * kMatmulSlices + i)};
        address += Products * s.accumulator_bytes;
    }
    s.output_start = address - s.uram;
    s.output_bytes = OutputBytes == sizeof(uint16_t) ? (uint64_t) s.slice_m * s.c_columns * OutputBytes : 0;
    address += kMatmulSlices * s.output_bytes;
    s.scratch = s.physical + address - s.uram;
    DAYAN_PARAMS_IF(!s.physical || address - s.uram + InputScratchBytes > dayan_uram_size() || address + InputScratchBytes > Constants::kUramWindowStride) {
        return -12;
    }
    if constexpr (Cores == 4) {
        cluster->uram_base[lane] = s.physical;
        matmul_check(dayan_cluster_barrier(4, &cluster->barrier));
    }
    const int grid = Cores == 4 && !Inference ? 2 : 1;
    const uint64_t columns = Inference ? (config->n + Cores - 1) / Cores : config->n;
    s.nt = (int) ((columns + grid * s.tile_n - 1) / (grid * s.tile_n));
    s.kt = (int) ((config->k + Constants::kTileK - 1) / Constants::kTileK);
    const uint64_t outputs = (matrix.m + grid * s.tile_m - 1) / (grid * s.tile_m) * s.nt;
    if (outputs > UINT64_MAX / s.kt) {
        return -22;
    }
    s.generations = outputs * s.kt;
    for (uint64_t i3 = 0; i3 < ne3; i3 += group3) {
        for (uint64_t i2 = 0; i2 < ne2; i2 += group2) {
            const uint64_t a_offset = i3 * config->a_batch_stride3 + i2 * config->a_batch_stride2;
            const uint64_t b_offset = (i3 / ratio3) * config->bt_batch_stride3 + (i2 / ratio2) * config->bt_batch_stride2;
            s.input[0] = (uintptr_t) a + a_offset;
            s.input[1] = (uintptr_t) bt0 + b_offset;
            s.input[2] = (uintptr_t) bt1 + b_offset;
            s.output = (uintptr_t) output + i3 * config->output_batch_stride3 + i2 * config->output_batch_stride2;
            MatmulLoadArgs loads[] = {{&s, false}, {&s, true}};
            dayan_thread_resources_t resources = {};
            resources.lsu_tcsr_count = InputLoader != nullptr ? 2 : 1;
            resources.vec_tcsr_count = InputLoader != nullptr ? 1 : 0;
            resources.local_resource_count = 12;
            resources.shared_resource_count = Cores == 4 ? 12 : 0;
            matmul_check(dayan_create_thread("load-a", matmul_load<Cores, Products, InputLoader>, &loads[0], &resources));
            resources.lsu_tcsr_count = 1;
            resources.vec_tcsr_count = 0;
            matmul_check(dayan_create_thread("load-b", matmul_load<Cores, Products, nullptr>, &loads[1], &resources));
            resources.vec_tcsr_count = HasEpilogue ? 1 : 0;
            matmul_check(dayan_create_thread("post", matmul_post<Products, OutputBytes, HasEpilogue, Epilogue>, &s, &resources));
            matmul_calc<Format, Products>(s);
            matmul_check(dayan_thread_join_all());
            if constexpr (Cores == 4) {
                matmul_check(dayan_cluster_barrier(4, &cluster->barrier));
                // Save phases by token so the next matrix can change operand orientation.
                for (int b = 0; b < 2; ++b) {
                    if (s.split_n && (b || lane != 0)) {
                        continue;
                    }
                    const bool buffered = b ? !s.buffer_a : s.buffer_a;
                    const int pair = s.split_n ? 0 : b ? lane & 1 : lane >> 1;
                    if (s.split_n || (b ? lane >> 1 : lane & 1) == 0) {
                        for (int i = 0; i < (buffered ? 2 : kMatmulSlices); ++i) {
                            const int index = buffered ? 2 * kMatmulSlices + pair * 2 + i : pair * kMatmulSlices + i;
                            cluster->shared_phase[index] = (s.load_phases[b] >> DAYAN_SHARED_RESOURCE(index)) & 1;
                        }
                    }
                }
                matmul_check(dayan_cluster_barrier(4, &cluster->barrier));
            }
        }
    }
    return 0;
}

} // namespace ggml_dayan_matmul_detail

#endif
