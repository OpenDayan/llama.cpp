#pragma once

#include "ggml-dayan-activation.h"

#if defined(__DAYAN_DEVICE__)

#    include <dayan_intrinsics.h>
#    include <dayan_sys.h>

namespace ggml_dayan_matmul_detail {

using ggml_dayan_vec_detail::begin_vec_block;
using ggml_dayan_vec_detail::configure_vec_dims;
using ggml_dayan_vec_detail::configure_vec_iterator;
using ggml_dayan_vec_detail::start_vec_block;

static constexpr long kVecSync   = ggml_dayan_vec_detail::kVecSync;
static constexpr long kVecV0Base = ggml_dayan_vec_detail::kVecV0Base;
static constexpr long kVecV1Base = ggml_dayan_vec_detail::kVecV1Base;
static constexpr long kVecV2Base = ggml_dayan_vec_detail::kVecV2Base;
static constexpr long kVecV3Base = ggml_dayan_vec_detail::kVecV3Base;

struct Constants {
    enum : int {
        kLsuLocal            = 0,
        kLsuShared           = 1,
        kLsuOutput           = 3,
        kTileM               = 128,
        kTileN               = 384,
        kTileK               = 256,
        kUramAlignment       = 32,
        kUramWindowStride    = 2 * 1024 * 1024,
        kVecLanes            = 64,
        kVecBytes            = kVecLanes * sizeof(float),
        kConstantStride      = 32,
        kConstantNegLog2E    = 0,
        kConstantOne         = 1,
        kConstantGeluA       = 2,
        kConstantSqrt2OverPi = 3,
        kConstantNeg2Log2E   = 4,
        kConstantTwo         = 5,
        kConstantHalf        = 6,
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

static_assert(MATMUL_DAYAN_1C_SLICES == 1 || MATMUL_DAYAN_1C_SLICES == 3, "MATMUL_DAYAN_1C_SLICES must be 1 or 3");
static_assert((Constants::kUramWindowStride & (Constants::kUramWindowStride - 1)) == 0,
              "URAM window stride must be a power of two");

static constexpr uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static inline int min_int(int left, int right) {
    return left < right ? left : right;
}

static inline uint64_t read_cycle() {
    uint64_t cycle;

    __asm__ volatile("csrr %0, mcycle" : "=r"(cycle) : : "memory");
    return cycle;
}

template <bool ProducerLoadsA, int AccumulatorCount, int OutputElementBytes, bool HasEpilogue> struct UramLayout {
    static constexpr int slice_offset(int slice) {
        return (ProducerLoadsA ? Constants::kTileN : Constants::kTileM) * slice / MATMUL_DAYAN_1C_SLICES;
    }

    static constexpr int slice_size(int slice) { return slice_offset(slice + 1) - slice_offset(slice); }

    static constexpr int max_rows(int slice) { return ProducerLoadsA ? Constants::kTileM : slice_size(slice); }

    static constexpr int max_cols(int slice) { return ProducerLoadsA ? slice_size(slice) : Constants::kTileN; }

    static constexpr uint64_t local_bytes(int slice) {
        return align_up((uint64_t) slice_size(slice) * Constants::kTileK * sizeof(uint16_t), Constants::kUramAlignment);
    }

    static constexpr uint64_t shared_bytes() {
        return align_up(
            (uint64_t) (ProducerLoadsA ? Constants::kTileM : Constants::kTileN) * Constants::kTileK * sizeof(uint16_t),
            Constants::kUramAlignment);
    }

    static constexpr uint64_t accumulator_bytes(int slice) {
        return align_up((uint64_t) max_rows(slice) * max_cols(slice) * sizeof(float), Constants::kUramAlignment);
    }

    static constexpr uint64_t output_bytes(int slice) {
        return HasEpilogue ? align_up((uint64_t) max_rows(slice) * max_cols(slice) * OutputElementBytes,
                                      Constants::kUramAlignment) :
                             0;
    }

    static constexpr uint64_t slice_bytes(int slice) {
        return align_up((uint64_t) AccumulatorCount * accumulator_bytes(slice) + output_bytes(slice),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t local_base(int slice) {
        uint64_t base = 0;

        for (int index = 0; index < slice; ++index) {
            base = align_up(base + local_bytes(index), Constants::kUramAlignment);
        }
        return base;
    }

    static constexpr uint64_t local_end() {
        return align_up(local_base(MATMUL_DAYAN_1C_SLICES - 1) + local_bytes(MATMUL_DAYAN_1C_SLICES - 1),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t shared_base(int buffer) {
        return align_up(local_end() + (uint64_t) buffer * shared_bytes(), Constants::kUramAlignment);
    }

    static constexpr uint64_t slice_start() {
        return align_up(shared_base(1) + shared_bytes(), Constants::kUramAlignment);
    }

    static constexpr uint64_t slice_base(int slice) {
        uint64_t base = slice_start();

        for (int index = 0; index < slice; ++index) {
            base = align_up(base + slice_bytes(index), Constants::kUramAlignment);
        }
        return base;
    }

    static constexpr uint64_t accumulator_base(int slice, int accumulator) {
        return slice_base(slice) + (uint64_t) accumulator * accumulator_bytes(slice);
    }

    static constexpr uint64_t output_base(int slice) {
        return slice_base(slice) + (uint64_t) AccumulatorCount * accumulator_bytes(slice);
    }

    static constexpr uint64_t constants_base() {
        return align_up(slice_base(MATMUL_DAYAN_1C_SLICES - 1) + slice_bytes(MATMUL_DAYAN_1C_SLICES - 1),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t total_bytes() {
        return align_up(constants_base() + (HasEpilogue ? Constants::kConstantCount * Constants::kConstantStride : 0),
                        Constants::kUramAlignment);
    }
};

struct SharedBuffer {
    dayan_resource_t  resource;
    volatile uint64_t generation;
    uint64_t          uram_offset;
};

struct MatmulState {
    int               m;
    int               n;
    int               k;
    int               count_m;
    int               count_n;
    int               count_k;
    uint64_t          a_base;
    uint64_t          bt_base[2];
    uint64_t          output_base;
    uint64_t          a_stride;
    uint64_t          bt_stride;
    uint64_t          output_stride;
    uint64_t          uram_address;
    uint64_t          uram_offset;
    uint64_t          a_swizzle;
    uint64_t          input_scratch;
    const void *      input_context;
    SharedBuffer      shared[2];
    volatile int      status;
    volatile uint32_t start;
    volatile uint32_t abort;
};

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
    uint64_t constants;
    int      rows;
    int      cols;
    int      row_stride;
};

using MatmulEpilogue = uint64_t (*)(const MatmulEpilogueArgs * args);

static inline int rows_for_tile(const MatmulState * state, int tile_m) {
    const int row_base = tile_m * Constants::kTileM;

    if (state->m <= row_base) {
        return 0;
    }
    return min_int(state->m - row_base, Constants::kTileM);
}

static inline int cols_for_tile(const MatmulState * state, int tile_n) {
    const int col_base = tile_n * Constants::kTileN;

    if (state->n <= col_base) {
        return 0;
    }
    return min_int(state->n - col_base, Constants::kTileN);
}

template <bool ProducerLoadsA, int AccumulatorCount, int OutputElementBytes, bool HasEpilogue>
static inline int rows_for_slice(const MatmulState * state, int tile_m, int slice) {
    using Layout = UramLayout<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>;

    if (ProducerLoadsA) {
        return rows_for_tile(state, tile_m);
    }
    const int row_base = tile_m * Constants::kTileM + Layout::slice_offset(slice);

    if (state->m <= row_base) {
        return 0;
    }
    return min_int(state->m - row_base, Layout::slice_size(slice));
}

template <bool ProducerLoadsA, int AccumulatorCount, int OutputElementBytes, bool HasEpilogue>
static inline int cols_for_slice(const MatmulState * state, int tile_n, int slice) {
    using Layout = UramLayout<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>;

    if (!ProducerLoadsA) {
        return cols_for_tile(state, tile_n);
    }
    const int col_base = tile_n * Constants::kTileN + Layout::slice_offset(slice);

    if (state->n <= col_base) {
        return 0;
    }
    return min_int(state->n - col_base, Layout::slice_size(slice));
}

template <bool ProducerLoadsA, int AccumulatorCount, int OutputElementBytes, bool HasEpilogue>
static inline int active_slices(const MatmulState * state, int tile_m, int tile_n) {
    int count = 0;

    for (int slice = 0; slice < MATMUL_DAYAN_1C_SLICES; ++slice) {
        if (rows_for_slice<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>(state, tile_m, slice) >
                0 &&
            cols_for_slice<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>(state, tile_n, slice) >
                0) {
            ++count;
        }
    }
    return count;
}

static inline int k_for_tile(const MatmulState * state, int tile_k) {
    const int k_base = tile_k * Constants::kTileK;

    if (state->k <= k_base) {
        return 0;
    }
    return min_int(state->k - k_base, Constants::kTileK);
}

static inline int remember_error(MatmulState * state, int status) {
    if (status < 0 && state->status == 0) {
        state->status = status;
    }
    return status;
}

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

static inline int wait_for_generation(SharedBuffer * buffer, uint64_t expected_generation) {
    for (;;) {
        if (buffer->generation == expected_generation) {
            return dayan_resource_wait_ready(&buffer->resource);
        }

        const int status = dayan_resource_wait_free(&buffer->resource);
        if (status < 0) {
            return status;
        }
        dayan_yield();
    }
}

static inline uint64_t constant_address(const MatmulEpilogueArgs * args, int constant) {
    return args->constants + (uint64_t) constant * Constants::kConstantStride;
}

static inline ggml_dayan_vec_detail::ActivationConstants activation_constants(const MatmulEpilogueArgs * args) {
    return {
        constant_address(args, Constants::kConstantNegLog2E),
        constant_address(args, Constants::kConstantOne),
        constant_address(args, Constants::kConstantGeluA),
        constant_address(args, Constants::kConstantSqrt2OverPi),
        constant_address(args, Constants::kConstantNeg2Log2E),
        constant_address(args, Constants::kConstantTwo),
        constant_address(args, Constants::kConstantHalf),
    };
}

static inline void issue_swiglu(const MatmulEpilogueArgs * args) {
    const long stride = args->row_stride * (int) sizeof(float);
    ggml_dayan_vec_detail::issue_swiglu(
        { args->accumulator[0], stride }, { args->accumulator[1], stride }, { args->accumulator[0], stride },
        activation_constants(args), args->rows, args->cols);
}

static inline void issue_geglu(const MatmulEpilogueArgs * args) {
    const long stride = args->row_stride * (int) sizeof(float);
    ggml_dayan_vec_detail::issue_geglu(
        { args->accumulator[0], stride }, { args->accumulator[1], stride }, { args->accumulator[0], stride },
        activation_constants(args), args->rows, args->cols);
}

template <long MatDataType, long LsuGroup> static inline void quantize(const MatmulEpilogueArgs * args) {
    const ggml_dayan_vec_detail::F32Tensor accumulator = {
        args->accumulator[0], args->row_stride * (int) sizeof(float)
    };
    ggml_dayan_vec_detail::convert_f32_to_lowp<MatDataType, LsuGroup>(
        accumulator, accumulator,
        { args->output, args->row_stride * (int) sizeof(uint16_t) }, args->rows, args->cols);
}

template <long GroupBase> static uint64_t epilogue_swiglu_bf16(const MatmulEpilogueArgs * args) {
    issue_swiglu(args);
    dayan_wait_vec();
    quantize<BaseNPUInstBuilder::DT_BF16, GroupBase + Constants::kLsuLocal>(args);
    return args->output;
}

template <long GroupBase> static uint64_t epilogue_swiglu_fp16(const MatmulEpilogueArgs * args) {
    issue_swiglu(args);
    dayan_wait_vec();
    quantize<BaseNPUInstBuilder::DT_FP16, GroupBase + Constants::kLsuLocal>(args);
    return args->output;
}

template <long GroupBase> static uint64_t epilogue_geglu_bf16(const MatmulEpilogueArgs * args) {
    issue_geglu(args);
    dayan_wait_vec();
    quantize<BaseNPUInstBuilder::DT_BF16, GroupBase + Constants::kLsuLocal>(args);
    return args->output;
}

template <long GroupBase> static uint64_t epilogue_geglu_fp16(const MatmulEpilogueArgs * args) {
    issue_geglu(args);
    dayan_wait_vec();
    quantize<BaseNPUInstBuilder::DT_FP16, GroupBase + Constants::kLsuLocal>(args);
    return args->output;
}

template <long GroupBase> static uint64_t epilogue_identity_bf16(const MatmulEpilogueArgs * args) {
    quantize<BaseNPUInstBuilder::DT_BF16, GroupBase + Constants::kLsuLocal>(args);
    return args->output;
}

template <long GroupBase> static uint64_t epilogue_identity_fp16(const MatmulEpilogueArgs * args) {
    quantize<BaseNPUInstBuilder::DT_FP16, GroupBase + Constants::kLsuLocal>(args);
    return args->output;
}

static uint64_t epilogue_identity(const MatmulEpilogueArgs * args) {
    return args->accumulator[0];
}

template <typename Format,
          bool ProducerLoadsA,
          long GroupBase,
          int  Slice,
          int  AccumulatorCount,
          int  OutputElementBytes,
          bool HasEpilogue>
static inline void configure_consumer(const MatmulState * state) {
    using Layout = UramLayout<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>;

    long               checksum   = 0;
    constexpr int      rows       = Layout::max_rows(Slice);
    constexpr int      cols       = Layout::max_cols(Slice);
    constexpr int      local_rows = Layout::slice_size(Slice);
    constexpr uint64_t swizzle    = MAT_BATCH_SWIZZLE(5, 3, 5);
    constexpr uint64_t c_swizzle  = HasEpilogue ? 0 : swizzle;

    set_lsu_matrix<GroupBase + Constants::kLsuLocal>(checksum, local_rows, Constants::kTileK * (int) sizeof(uint16_t),
                                                     (long) (ProducerLoadsA ? state->bt_stride : state->a_stride),
                                                     Constants::kTileK * (int) sizeof(uint16_t));
    set_lsu_matrix<GroupBase + Constants::kLsuOutput>(checksum, rows, cols * OutputElementBytes,
                                                      cols * OutputElementBytes, (long) state->output_stride);
    __tx_lsu_tcsr_write<(GroupBase + Constants::kLsuLocal) * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(swizzle);
    __tx_lsu_tcsr_write<(GroupBase + Constants::kLsuOutput) * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(c_swizzle);

    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(rows - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(cols - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(Constants::kTileK - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASTRIDE>(Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSTRIDE>(Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSTRIDE>(cols * sizeof(float));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(state->a_swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(c_swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_DATATYPE>(Format::kMatDataType);
}

template <bool ProducerLoadsA, int AccumulatorCount>
static inline uint64_t shared_generation(const MatmulState * state,
                                         int                 tile_m,
                                         int                 tile_n,
                                         int                 tile_k,
                                         int                 accumulator) {
    const uint64_t tile = ((uint64_t) tile_m * state->count_n + tile_n) * state->count_k + tile_k;
    return ProducerLoadsA ? tile : tile * AccumulatorCount + accumulator;
}

template <typename Format,
          bool           ProducerLoadsA,
          long           GroupBase,
          int            Slice,
          int            AccumulatorCount,
          int            OutputElementBytes,
          bool           HasEpilogue,
          MatmulEpilogue Epilogue>
static void consume_slice(MatmulState * state) {
    using Layout = UramLayout<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>;

    constexpr int  slice_offset           = Layout::slice_offset(Slice);
    constexpr int  max_rows               = Layout::max_rows(Slice);
    constexpr int  max_cols               = Layout::max_cols(Slice);
    constexpr int  local_rows             = Layout::slice_size(Slice);
    const uint64_t local_offset           = state->uram_offset + Layout::local_base(Slice);
    const uint64_t local_address          = state->uram_address + Layout::local_base(Slice);
    uint64_t       accumulator_offset[2]  = { 0, 0 };
    uint64_t       accumulator_address[2] = { 0, 0 };

    for (int accumulator = 0; accumulator < AccumulatorCount; ++accumulator) {
        accumulator_offset[accumulator]  = state->uram_offset + Layout::accumulator_base(Slice, accumulator);
        accumulator_address[accumulator] = state->uram_address + Layout::accumulator_base(Slice, accumulator);
    }

    while (!state->start) {
        dayan_yield();
    }
    if (state->abort) {
        return;
    }

    configure_consumer<Format, ProducerLoadsA, GroupBase, Slice, AccumulatorCount, OutputElementBytes, HasEpilogue>(
        state);

    for (int tile_m = 0; tile_m < state->count_m; ++tile_m) {
        const int real_rows =
            rows_for_slice<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>(state, tile_m, Slice);

        if (real_rows == 0) {
            continue;
        }
        for (int tile_n = 0; tile_n < state->count_n; ++tile_n) {
            const int real_cols =
                cols_for_slice<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>(state, tile_n, Slice);

            if (real_cols == 0) {
                continue;
            }

            for (int tile_k = 0; tile_k < state->count_k; ++tile_k) {
                const int  real_k  = k_for_tile(state, tile_k);
                const bool first_k = tile_k == 0;
                const bool last_k  = tile_k + 1 == state->count_k;

                if (!ProducerLoadsA) {
                    const uint64_t local_dram =
                        state->a_base + (uint64_t) tile_k * Constants::kTileK * sizeof(uint16_t) +
                        ((uint64_t) tile_m * Constants::kTileM + slice_offset) * state->a_stride;
                    load_matrix<GroupBase + Constants::kLsuLocal>((uint64_t) real_k * sizeof(uint16_t), real_rows,
                                                                  Constants::kTileK * sizeof(uint16_t), local_rows,
                                                                  local_address, local_dram);
                    dayan_wait_lsu();
                }

                for (int accumulator = 0; accumulator < AccumulatorCount; ++accumulator) {
                    if (ProducerLoadsA) {
                        const uint64_t local_dram =
                            state->bt_base[accumulator] + (uint64_t) tile_k * Constants::kTileK * sizeof(uint16_t) +
                            ((uint64_t) tile_n * Constants::kTileN + slice_offset) * state->bt_stride;
                        load_matrix<GroupBase + Constants::kLsuLocal>((uint64_t) real_k * sizeof(uint16_t), real_cols,
                                                                      Constants::kTileK * sizeof(uint16_t), local_rows,
                                                                      local_address, local_dram);
                        dayan_wait_lsu();
                    }

                    const uint64_t generation =
                        shared_generation<ProducerLoadsA, AccumulatorCount>(state, tile_m, tile_n, tile_k, accumulator);
                    SharedBuffer * shared = &state->shared[generation & 1U];
                    int            status = wait_for_generation(shared, generation);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }

                    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(real_rows - 1);
                    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(real_cols - 1);
                    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(real_k - 1);
                    if (ProducerLoadsA) {
                        if (first_k) {
                            __tx_mat_mm<GroupBase>((long) accumulator_offset[accumulator], (long) shared->uram_offset,
                                                   (long) local_offset);
                        } else {
                            __tx_mat_mma<GroupBase>((long) accumulator_offset[accumulator], (long) shared->uram_offset,
                                                    (long) local_offset);
                        }
                    } else {
                        if (first_k) {
                            __tx_mat_mm<GroupBase>((long) accumulator_offset[accumulator], (long) local_offset,
                                                   (long) shared->uram_offset);
                        } else {
                            __tx_mat_mma<GroupBase>((long) accumulator_offset[accumulator], (long) local_offset,
                                                    (long) shared->uram_offset);
                        }
                    }
                    dayan_wait_mat();

                    status = dayan_resource_consume(&shared->resource);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }
                }

                if (last_k) {
                    MatmulEpilogueArgs args;
                    args.accumulator[0]        = accumulator_address[0];
                    args.accumulator[1]        = AccumulatorCount > 1 ? accumulator_address[1] : 0;
                    args.output                = state->uram_address + Layout::output_base(Slice);
                    args.constants             = state->uram_address + Layout::constants_base();
                    args.rows                  = real_rows;
                    args.cols                  = real_cols;
                    args.row_stride            = max_cols;
                    const uint64_t output_uram = Epilogue(&args);
                    if constexpr (HasEpilogue) {
                        long checksum = 0;
                        constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);
                        set_lsu_matrix<GroupBase + Constants::kLsuLocal>(
                            checksum, local_rows, Constants::kTileK * (int) sizeof(uint16_t),
                            (long) (ProducerLoadsA ? state->bt_stride : state->a_stride),
                            Constants::kTileK * (int) sizeof(uint16_t));
                        __tx_lsu_tcsr_write<
                            (GroupBase + Constants::kLsuLocal) * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(swizzle);
                    }
                    const uint64_t output_dram =
                        state->output_base +
                        ((uint64_t) tile_n * Constants::kTileN + (ProducerLoadsA ? slice_offset : 0)) *
                            OutputElementBytes +
                        ((uint64_t) tile_m * Constants::kTileM + (ProducerLoadsA ? 0 : slice_offset)) *
                            state->output_stride;

                    store_matrix<GroupBase + Constants::kLsuOutput>((uint64_t) real_cols * OutputElementBytes,
                                                                    real_rows, max_cols * OutputElementBytes, max_rows,
                                                                    output_uram, output_dram);
                } else {
                    dayan_yield();
                }
            }
        }
    }
    dayan_wait_lsu();
}

template <typename Format,
          bool           ProducerLoadsA,
          int            AccumulatorCount,
          int            OutputElementBytes,
          bool           HasEpilogue,
          MatmulEpilogue Epilogue>
static void consumer_0(void * argument) {
    consume_slice<Format, ProducerLoadsA, 0, 0, AccumulatorCount, OutputElementBytes, HasEpilogue, Epilogue>(
        (MatmulState *) argument);
}

#    if MATMUL_DAYAN_1C_SLICES == 3
template <typename Format,
          bool           ProducerLoadsA,
          int            AccumulatorCount,
          int            OutputElementBytes,
          bool           HasEpilogue,
          MatmulEpilogue Epilogue>
static void consumer_1(void * argument) {
    consume_slice<Format, ProducerLoadsA, 4, 1, AccumulatorCount, OutputElementBytes, HasEpilogue, Epilogue>(
        (MatmulState *) argument);
}

template <typename Format,
          bool           ProducerLoadsA,
          int            AccumulatorCount,
          int            OutputElementBytes,
          bool           HasEpilogue,
          MatmulEpilogue Epilogue>
static void consumer_2(void * argument) {
    consume_slice<Format, ProducerLoadsA, 2, 2, AccumulatorCount, OutputElementBytes, HasEpilogue, Epilogue>(
        (MatmulState *) argument);
}
#    endif

template <bool ProducerLoadsA> static inline void configure_producer(const MatmulState * state) {
    long               checksum = 0;
    constexpr uint64_t swizzle  = MAT_BATCH_SWIZZLE(5, 3, 5);

    set_lsu_matrix<Constants::kLsuShared>(
        checksum, ProducerLoadsA ? Constants::kTileM : Constants::kTileN, Constants::kTileK * (int) sizeof(uint16_t),
        (long) (ProducerLoadsA ? state->a_stride : state->bt_stride), Constants::kTileK * (int) sizeof(uint16_t));
    __tx_lsu_tcsr_write<Constants::kLsuShared * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(swizzle);
}

template <bool              ProducerLoadsA,
          int               AccumulatorCount,
          int               OutputElementBytes,
          bool              HasEpilogue,
          MatmulInputLoader InputLoader,
          uint64_t          InputScratchBytes>
static int produce_shared(MatmulState * state) {
    configure_producer<ProducerLoadsA>(state);

    for (int tile_m = 0; tile_m < state->count_m; ++tile_m) {
        for (int tile_n = 0; tile_n < state->count_n; ++tile_n) {
            const int active =
                active_slices<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>(state, tile_m, tile_n);

            if (active == 0) {
                continue;
            }

            for (int tile_k = 0; tile_k < state->count_k; ++tile_k) {
                const int producer_count = ProducerLoadsA ? 1 : AccumulatorCount;

                for (int accumulator = 0; accumulator < producer_count; ++accumulator) {
                    const uint64_t generation =
                        shared_generation<ProducerLoadsA, AccumulatorCount>(state, tile_m, tile_n, tile_k, accumulator);
                    SharedBuffer * shared = &state->shared[generation & 1U];
                    const int      real_k = k_for_tile(state, tile_k);
                    const uint64_t shared_dram =
                        (ProducerLoadsA ? state->a_base : state->bt_base[accumulator]) +
                        (uint64_t) tile_k * Constants::kTileK * sizeof(uint16_t) +
                        (ProducerLoadsA ? (uint64_t) tile_m * Constants::kTileM * state->a_stride :
                                          (uint64_t) tile_n * Constants::kTileN * state->bt_stride);
                    int status = dayan_resource_wait_free(&shared->resource);

                    if (status < 0) {
                        return remember_error(state, status);
                    }
                    const int users = ProducerLoadsA ? active * AccumulatorCount : active;
                    status          = dayan_resource_alloc(&shared->resource, 1, (uint32_t) users);
                    if (status < 0) {
                        return remember_error(state, status);
                    }

                    shared->generation = generation;
                    if constexpr (ProducerLoadsA && InputLoader != nullptr) {
                        const MatmulInputArgs args = {
                            state->input_context,
                            state->uram_address + (shared->uram_offset - state->uram_offset),
                            state->input_scratch,
                            tile_m * Constants::kTileM,
                            tile_k * Constants::kTileK,
                            rows_for_tile(state, tile_m),
                            real_k,
                        };
                        status = InputLoader(&args);
                        if (status < 0) {
                            return remember_error(state, status);
                        }
                    } else {
                        load_matrix<Constants::kLsuShared>(
                            (uint64_t) real_k * sizeof(uint16_t),
                            ProducerLoadsA ? rows_for_tile(state, tile_m) : cols_for_tile(state, tile_n),
                            Constants::kTileK * sizeof(uint16_t),
                            ProducerLoadsA ? Constants::kTileM : Constants::kTileN,
                            state->uram_address + (shared->uram_offset - state->uram_offset), shared_dram);
                        dayan_wait_lsu();
                    }

                    status = dayan_resource_produce(&shared->resource);
                    if (status < 0) {
                        return remember_error(state, status);
                    }
                    dayan_yield();
                }
            }
        }
    }
    return 0;
}

template <typename Format,
          bool              ProducerLoadsA,
          int               AccumulatorCount,
          int               OutputElementBytes,
          bool              HasEpilogue,
          MatmulEpilogue    Epilogue0,
          MatmulEpilogue    Epilogue1,
          MatmulEpilogue    Epilogue2,
          MatmulInputLoader InputLoader,
          uint64_t          InputScratchBytes>
static int run_pipeline(const DayanMatmulConfig * config,
                        const uint16_t *          a,
                        const uint16_t *          bt0,
                        const uint16_t *          bt1,
                        void *                    output,
                        uint64_t                  uram_address,
                        uint64_t                  uram_offset,
                        uint64_t *                pipeline_begin,
                        const void *              input_context) {
    using Layout = UramLayout<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue>;

    MatmulState state;
    state.m             = (int) config->m;
    state.n             = (int) config->n;
    state.k             = (int) config->k;
    state.count_m       = (state.m + Constants::kTileM - 1) / Constants::kTileM;
    state.count_n       = (state.n + Constants::kTileN - 1) / Constants::kTileN;
    state.count_k       = (state.k + Constants::kTileK - 1) / Constants::kTileK;
    state.a_base        = (uint64_t) (uintptr_t) a;
    state.bt_base[0]    = (uint64_t) (uintptr_t) bt0;
    state.bt_base[1]    = (uint64_t) (uintptr_t) bt1;
    state.output_base   = (uint64_t) (uintptr_t) output;
    state.a_stride      = config->a_stride;
    state.bt_stride     = config->bt_stride;
    state.output_stride = config->output_stride;
    state.uram_address  = uram_address;
    state.uram_offset   = uram_offset;
    state.a_swizzle     = InputLoader == nullptr ? MAT_BATCH_SWIZZLE(5, 3, 5) : 0;
    state.input_scratch = uram_address + Layout::total_bytes();
    state.input_context = input_context;
    state.status        = 0;
    state.start         = 0;
    state.abort         = 0;

    for (int index = 0; index < 2; ++index) {
        int status = dayan_resource_init(&state.shared[index].resource);

        if (status < 0) {
            return status;
        }
        state.shared[index].generation  = ~0ULL;
        state.shared[index].uram_offset = state.uram_offset + Layout::shared_base(index);
    }

    int status = dayan_create_proc(
        "consumer-0", consumer_0<Format, ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue, Epilogue0>,
        &state);
#    if MATMUL_DAYAN_1C_SLICES == 3
    if (status >= 0) {
        status = dayan_create_proc(
            "consumer-1",
            consumer_1<Format, ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue, Epilogue1>, &state);
    }
    if (status >= 0) {
        status = dayan_create_proc(
            "consumer-2",
            consumer_2<Format, ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue, Epilogue2>, &state);
    }
#    endif
    if (status < 0) {
        state.status = status;
        state.abort  = 1;
        state.start  = 1;
        (void) dayan_join_all();
        return status;
    }

    *pipeline_begin = read_cycle();
    state.start     = 1;
    status          = produce_shared<ProducerLoadsA, AccumulatorCount, OutputElementBytes, HasEpilogue, InputLoader,
                                     InputScratchBytes>(&state);
    if (status < 0) {
        state.status = status;
    }

    status = dayan_join_all();
    if (status < 0 && state.status == 0) {
        state.status = status;
    }
    return state.status;
}

template <int AccumulatorCount, int OutputElementBytes>
static inline int config_valid(const DayanMatmulConfig * config) {
    const uint64_t int_limit        = 0x7ffff000ULL;
    const uint64_t output_alignment = OutputElementBytes == 4 ? 64U : 32U;

    if (!config || !config->m || !config->n || !config->k || config->m > int_limit || config->n > int_limit ||
        config->k > int_limit || (config->n % 16U) != 0 || config->a_stride < config->k * sizeof(uint16_t) ||
        config->bt_stride < config->k * sizeof(uint16_t) || config->output_stride < config->n * OutputElementBytes ||
        (config->output_stride & (output_alignment - 1U)) || config->a_stride > 0x7fffffffffffffffULL ||
        config->bt_stride > 0x7fffffffffffffffULL || config->output_stride > 0x7fffffffffffffffULL) {
        return 0;
    }
    return AccumulatorCount == 1 || AccumulatorCount == 2;
}

template <bool HasEpilogue, typename Layout> static inline void initialize_constants(uint64_t uram_address) {
    if (!HasEpilogue) {
        return;
    }

    const uint64_t base = uram_address + Layout::constants_base();
    *(volatile float *) (uintptr_t) (base + Constants::kConstantNegLog2E * Constants::kConstantStride) =
        -1.4426950408889634f;
    *(volatile float *) (uintptr_t) (base + Constants::kConstantOne * Constants::kConstantStride)   = 1.0f;
    *(volatile float *) (uintptr_t) (base + Constants::kConstantGeluA * Constants::kConstantStride) = 0.044715f;
    *(volatile float *) (uintptr_t) (base + Constants::kConstantSqrt2OverPi * Constants::kConstantStride) =
        0.7978845608028654f;
    *(volatile float *) (uintptr_t) (base + Constants::kConstantNeg2Log2E * Constants::kConstantStride) =
        -2.8853900817779268f;
    *(volatile float *) (uintptr_t) (base + Constants::kConstantTwo * Constants::kConstantStride)  = 2.0f;
    *(volatile float *) (uintptr_t) (base + Constants::kConstantHalf * Constants::kConstantStride) = 0.5f;
}

template <typename Format,
          int               AccumulatorCount,
          int               OutputElementBytes,
          bool              HasEpilogue,
          MatmulEpilogue    AProducerEpilogue0,
          MatmulEpilogue    AProducerEpilogue1,
          MatmulEpilogue    AProducerEpilogue2,
          MatmulEpilogue    BProducerEpilogue0,
          MatmulEpilogue    BProducerEpilogue1,
          MatmulEpilogue    BProducerEpilogue2,
          MatmulInputLoader InputLoader       = nullptr,
          uint64_t          InputScratchBytes = 0>
static void run_matmul_kernel(const DayanMatmulConfig * config,
                              const uint16_t *          a,
                              const uint16_t *          bt0,
                              const uint16_t *          bt1,
                              void *                    output,
                              DayanMatmulPerf *         perf,
                              const void *              input_context = nullptr) {
    using AProducerLayout = UramLayout<true, AccumulatorCount, OutputElementBytes, HasEpilogue>;
    using BProducerLayout = UramLayout<false, AccumulatorCount, OutputElementBytes, HasEpilogue>;

    static_assert(AProducerLayout::total_bytes() <= MATMUL_DAYAN_1C_URAM_BYTES,
                  "A-producer matmul URAM layout is too large");
    static_assert(AProducerLayout::total_bytes() + InputScratchBytes <= MATMUL_DAYAN_1C_URAM_BYTES,
                  "A-producer matmul input scratch is too large");
    static_assert(BProducerLayout::total_bytes() <= MATMUL_DAYAN_1C_URAM_BYTES,
                  "B-producer matmul URAM layout is too large");

    const uint64_t kernel_begin = read_cycle();

    if (!config_valid<AccumulatorCount, OutputElementBytes>(config) || !a || !bt0 || !output || !perf ||
        (AccumulatorCount == 2 && !bt1) || (InputLoader != nullptr && config != nullptr && config->m >= config->n) ||
        dayan_task_mode() != 1U || dayan_task_dim(DAYAN_TASK_X) != 1U || dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        DAYAN_PARAMS_ERROR();
    }

    void *         uram             = dayan_uram_base();
    const bool     producer_loads_a = config->m < config->n;
    const uint64_t required_uram =
        (producer_loads_a ? AProducerLayout::total_bytes() : BProducerLayout::total_bytes()) + InputScratchBytes;
    if (!uram || dayan_uram_size() < required_uram) {
        dayan_proc_exit(-12);
    }

    const uint64_t uram_address   = (uint64_t) (uintptr_t) uram;
    const uint64_t uram_offset    = uram_address & (Constants::kUramWindowStride - 1U);
    uint64_t       pipeline_begin = 0;
    int            status;

    if (producer_loads_a) {
        initialize_constants<HasEpilogue, AProducerLayout>(uram_address);
        status = run_pipeline<Format, true, AccumulatorCount, OutputElementBytes, HasEpilogue, AProducerEpilogue0,
                              AProducerEpilogue1, AProducerEpilogue2, InputLoader, InputScratchBytes>(
            config, a, bt0, bt1, output, uram_address, uram_offset, &pipeline_begin, input_context);
    } else {
        initialize_constants<HasEpilogue, BProducerLayout>(uram_address);
        status = run_pipeline<Format, false, AccumulatorCount, OutputElementBytes, HasEpilogue, BProducerEpilogue0,
                              BProducerEpilogue1, BProducerEpilogue2, InputLoader, InputScratchBytes>(
            config, a, bt0, bt1, output, uram_address, uram_offset, &pipeline_begin, input_context);
    }
    if (status < 0) {
        dayan_proc_exit(status);
    }

    const uint64_t pipeline_end = read_cycle();
    const uint64_t kernel_end   = read_cycle();

    DayanMatmulPerf * perf_uram = (DayanMatmulPerf *) (uintptr_t) uram_address;
    perf_uram->magic            = kDayanMatmulPerfMagic;
    perf_uram->m                = config->m;
    perf_uram->n                = config->n;
    perf_uram->k                = config->k;
    perf_uram->kernel_cycles    = kernel_end - kernel_begin;
    perf_uram->pipeline_cycles  = pipeline_end - pipeline_begin;

    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, 64, 64, 64);
    store_matrix<0>(sizeof(*perf_uram), 1, 64, 1, uram_address, (uint64_t) (uintptr_t) perf);
    dayan_wait_lsu();
}

}  // namespace ggml_dayan_matmul_detail

#endif
