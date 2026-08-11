#pragma once

// Single-core FlashAttention for contiguous BF16 Q/K/V tensors.
//
// A shared KV producer feeds three fixed Q-slice consumers, following the coroutine pipeline in
// kernel_bf16_matmul_os.cpp.  The two matrix products use BF16 inputs and accumulate to FP32:
//   S  = Q_bf16 * K_bf16^T
//   PV = P_bf16 * V_bf16
// Online-softmax state (m, l, O) stays in FP32.  Scores are never materialized in DRAM.

#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_macro_csr.hpp"
#include "npu_mat_csr.hpp"
#include "npu_task.hpp"

#include <stdint.h>

namespace {

constexpr long kConsumerCount = 3;
constexpr long kLsuK          = 0;
constexpr long kLsuVTranspose = 1;

constexpr uint32_t kFlagCausal = 1u;
constexpr long     kBr         = 64;
constexpr long     kBc         = 64;
constexpr long     kMaxHeadDim = 256;
constexpr long     kVecLanes   = 64;
constexpr long     kVecBytes   = kVecLanes * sizeof(float);
constexpr long     kUramBytes  = 1024 * 1024;

// Fixed-size buffers use the maximum supported head dimension.  Runtime strides inside Q/K,
// Vt, PV, and O use the actual head dimension.
constexpr long kQBytes        = kBr * kMaxHeadDim * sizeof(uint16_t);
constexpr long kKBytes        = kBc * kMaxHeadDim * sizeof(uint16_t);
constexpr long kVtBytes       = kMaxHeadDim * kBc * sizeof(uint16_t);
constexpr long kScoreBytes    = kBr * kBc * sizeof(float);
constexpr long kPBf16Bytes    = kBr * kBc * sizeof(uint16_t);
constexpr long kPVBytes       = kBr * kMaxHeadDim * sizeof(float);
constexpr long kOBytes        = kBr * kMaxHeadDim * sizeof(float);
constexpr long kRowStateBytes = kBr * kVecBytes;

// Each compute context owns every writable buffer touched by its Q tile.  Three 280-KiB
// contexts leave enough room for a shared double-buffered K/Vt pipeline and read-only vectors.
constexpr long kCtxQ         = 0x00000;
constexpr long kCtxScore     = 0x08000;
constexpr long kCtxPBf16     = 0x0c000;
constexpr long kCtxPV        = 0x0e000;
constexpr long kCtxO         = 0x1e000;
constexpr long kCtxM         = 0x2e000;
constexpr long kCtxL         = 0x32000;
constexpr long kCtxNewM      = 0x36000;
constexpr long kCtxTileSum   = 0x3a000;
constexpr long kCtxAlpha     = 0x3e000;
constexpr long kCtxThreshold = 0x42000;
constexpr long kContextBytes = 0x46000;

constexpr long context_base(long consumer) {
    return consumer * kContextBytes;
}

constexpr long kUramK0    = context_base(kConsumerCount);
constexpr long kUramVt0   = kUramK0 + kKBytes;
constexpr long kUramK1    = kUramVt0 + kVtBytes;
constexpr long kUramVt1   = kUramK1 + kKBytes;
constexpr long kUramLane  = kUramVt1 + kVtBytes;
constexpr long kUramScale = kUramLane + kVecBytes;
constexpr long kUramFloor = kUramScale + kVecBytes;
constexpr long kUramZero  = kUramFloor + kVecBytes;
constexpr long kUramEnd   = kUramZero + kVecBytes;

static_assert(kCtxScore == kCtxQ + kQBytes && kCtxPBf16 == kCtxScore + kScoreBytes &&
                  kCtxPV == kCtxPBf16 + kPBf16Bytes && kCtxO == kCtxPV + kPVBytes && kCtxM == kCtxO + kOBytes &&
                  kCtxL == kCtxM + kRowStateBytes && kCtxNewM == kCtxL + kRowStateBytes &&
                  kCtxTileSum == kCtxNewM + kRowStateBytes && kCtxAlpha == kCtxTileSum + kRowStateBytes &&
                  kCtxThreshold == kCtxAlpha + kRowStateBytes && kContextBytes == kCtxThreshold + kRowStateBytes,
              "FlashAttention compute-context layout is inconsistent");
static_assert(kUramK0 == 0xd2000 && kUramEnd == 0xf2400 && kUramEnd <= kUramBytes,
              "FlashAttention URAM layout exceeds 1 MiB");

template <typename T> static inline T * uram_ptr(long byte_offset) {
    return reinterpret_cast<T *>(static_cast<char *>(dayan_uram_base()) + byte_offset);
}

static inline uint64_t uram_addr(long byte_offset) {
    return reinterpret_cast<uint64_t>(static_cast<char *>(dayan_uram_base()) + byte_offset);
}

static inline long min_long(long a, long b) {
    return a < b ? a : b;
}

static inline long kv_k_base(long buffer_id) {
    return buffer_id == 0 ? kUramK0 : kUramK1;
}

// Copy a row-major matrix whose row size is a multiple of 64 bytes.  BlockTranspose=6 is the
// defined no-transpose setting and permits arbitrary base alignment.
template <long GROUP>
static inline void configure_lsu_rows(long rows, long row_bytes, long input_stride, long output_stride) {
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM0>(row_bytes / 64 - 1);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM1>(rows - 1);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_stride);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_stride);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

// Transpose one at-most-32-row BF16 panel from [rows, D] in DRAM into its corresponding
// 32-column slice of [D, 64] in URAM.  BlockPadding makes a short panel exactly 32 rows and
// fills the missing K positions with zero before the 32x32 transpose.
template <long GROUP>
static inline void load_v_panel_transposed(uint64_t v_dram,
                                           uint64_t vt_uram,
                                           long     rows,
                                           long     head_dim,
                                           uint64_t row_stride) {
    constexpr long side      = 32;
    constexpr long vt_stride = kBc * sizeof(uint16_t);

    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCK>(rows - 1);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM0>(head_dim / side - 1);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(row_stride);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(vt_stride);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(vt_stride * side);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(1);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKPADDING>(side - 1);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
    __tx_lsu_dma_d2u<GROUP>(vt_uram, v_dram);
}

static inline void load_v_transposed(uint64_t v_dram, long vt_base, long rows, long head_dim, uint64_t row_stride) {
    // A single short panel populates K positions [0, 31].  The second half is not touched by
    // that DMA and must not retain values from the preceding key tile.  Clear the strided 64-byte
    // half-rows with one U2U DMA instead of head_dim*32 scalar BF16 stores.
    if (rows <= 32) {
        configure_lsu_rows<kLsuVTranspose>(head_dim, 64, 0, kBc * sizeof(uint16_t));
        __tx_lsu_dma_u2u<kLsuVTranspose>(uram_addr(vt_base) + 32 * sizeof(uint16_t), uram_addr(kUramZero));
    }

    const long first_rows = min_long(rows, 32);
    load_v_panel_transposed<kLsuVTranspose>(v_dram, uram_addr(vt_base), first_rows, head_dim, row_stride);
    if (rows > 32) {
        load_v_panel_transposed<kLsuVTranspose>(v_dram + 32 * row_stride, uram_addr(vt_base) + 32 * sizeof(uint16_t),
                                                rows - 32, head_dim, row_stride);
    }
}

template <long GROUP>
static inline void configure_mat(long m, long n, long k, long a_stride, long b_stride, long c_stride) {
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_M>(m - 1);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_N>(n - 1);
    // Unlike kernel_bf16_matmul_os.cpp, K is the actual inner dimension, not a fixed 256.
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_K>(k - 1);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_ASTRIDE>(a_stride);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_BSTRIDE>(b_stride);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_CSTRIDE>(c_stride);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<GROUP * MAT_GROUPSZ + MAT_DATATYPE>(BaseNPUInstBuilder::DT_BF16);
}

template <long GROUP> static inline void issue_qk(long ctx_base, long rows, long cols, long head_dim, long k_base) {
    configure_mat<GROUP>(rows, cols, head_dim, head_dim * sizeof(uint16_t), head_dim * sizeof(uint16_t),
                         kBc * sizeof(float));
    // MAT addresses are 20-bit URAM offsets.  Unlike LSU, MACRO, and VEC, the MAT unit does not
    // normalize the mapped URAM address before indexing its local memory.
    // MM overwrites the score tile, so kCtxScore must not be initialized beforehand.
    __tx_mat_mm<GROUP>(ctx_base + kCtxScore, ctx_base + kCtxQ, k_base);
}

template <long GROUP> static inline void issue_p_to_bf16(long ctx_base, long rows) {
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM0>(kBc - 1);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM1>(rows - 1);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(kBc * sizeof(float));
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(kBc * sizeof(uint16_t));
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(0);
    __tx_macro_convert_fp32_bf16<GROUP>(reinterpret_cast<void *>(uram_addr(ctx_base + kCtxPBf16)),
                                        reinterpret_cast<void *>(uram_addr(ctx_base + kCtxScore)));
}

template <long GROUP> static inline void issue_pv(long ctx_base, long rows, long head_dim, long vt_base) {
    configure_mat<GROUP>(rows, head_dim, kBc, kBc * sizeof(uint16_t), kBc * sizeof(uint16_t), head_dim * sizeof(float));
    // MM overwrites the PV tile; only O is an online accumulator and needs an initial value.
    __tx_mat_mm<GROUP>(ctx_base + kCtxPV, ctx_base + kCtxPBf16, vt_base);
}

// VEC uses the linear TCSR layout documented by the current VEC specification.  The older
// include/npu_vec_csr.hpp describes a different layout and must not be used here.
enum VecTcsr {
    VEC_CONTROL = 0,
    VEC_DIM0    = 2,
    VEC_DIM1    = 3,
    VEC_DIM2    = 4,
    VEC_DIM3    = 5,
    VEC_DIM4    = 6,
    VEC_DIM5    = 7,
    VEC_V0_BASE = 37,
    VEC_V1_BASE = 44,
    VEC_V2_BASE = 51,
    VEC_V3_BASE = 58,
};

template <long BASE>
static inline void configure_vec_iterator(long dim0_stride,
                                          long dim1_stride,
                                          long dim2_stride,
                                          long dim3_stride,
                                          long dim4_stride = 0,
                                          long dim5_stride = 0) {
    __tx_vec_tcsr_write<BASE>(0);
    __tx_vec_tcsr_write<BASE + 1>(dim0_stride);
    __tx_vec_tcsr_write<BASE + 2>(dim1_stride);
    __tx_vec_tcsr_write<BASE + 3>(dim2_stride);
    __tx_vec_tcsr_write<BASE + 4>(dim3_stride);
    __tx_vec_tcsr_write<BASE + 5>(dim4_stride);
    __tx_vec_tcsr_write<BASE + 6>(dim5_stride);
}

static inline void configure_vec_dims(long dim2, long dim3 = 1) {
    __tx_vec_tcsr_write<VEC_DIM0>(0);
    __tx_vec_tcsr_write<VEC_DIM1>(0);
    __tx_vec_tcsr_write<VEC_DIM2>(dim2 - 1);
    __tx_vec_tcsr_write<VEC_DIM3>(dim3 - 1);
    __tx_vec_tcsr_write<VEC_DIM4>(0);
    __tx_vec_tcsr_write<VEC_DIM5>(0);
}

static inline long vec_control(long instructions) {
    const uint64_t n = static_cast<uint64_t>(instructions) & 0xff;
    return static_cast<long>(n | (n << 8) | (n << 16) | (n << 24) | (n << 32));
}

static inline void begin_vec_block(long instructions) {
    long free_entries;
    do {
        __tx_vec_tcsr_read<VEC_CONTROL>(free_entries);
    } while (free_entries < instructions);
    // Control snapshots the complete iterator/TCSR configuration into the allocated block slot.
    // Do not VEC_SYNC here: it would drain all earlier blocks and serialize the three consumers.
    // Cooperative tasks cannot yield between this check and their Control write.
}

static inline void start_vec_block(long instructions) {
    __tx_vec_tcsr_write<VEC_CONTROL>(vec_control(instructions));
}

static inline void emit_max_stage(long xor_mask) {
    // Q0 latest is x.  Shuffle writes Q1; cmp writes predicate s0; mux writes max(x, peer) to Q0.
    __tx_vec_shuffle_xor<1, 0>(xor_mask);
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<0, 16, 0, 0>();
}

static inline void emit_sum_stage(long xor_mask) {
    __tx_vec_shuffle_xor<1, 0>(xor_mask);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
}

constexpr long kMaskedSoftmaxVecInstructions = 73;
constexpr long kFullSoftmaxVecInstructions   = 63;

// One VEC thread handles one score row.  Values crossing logical phases are spilled to aligned
// URAM rows so no live Clock-hands value relies on an age near the queue-depth limit.
// Full 64-column tiles omit both lane-mask sequences, saving ten instructions on the common path.
template <bool MASK_SCORES>
static inline void issue_online_softmax_impl(long ctx_base, long rows, long threshold_row_stride) {
    configure_vec_dims(rows);
    configure_vec_iterator<VEC_V0_BASE>(0, 0, kVecBytes, 0);  // score/P row
    configure_vec_iterator<VEC_V1_BASE>(0, 0, 0, 0);          // shared constants
    if constexpr (MASK_SCORES) {
        // broadcast32 expands one compact int32 threshold to all 64 lanes.  A KV tail reuses one
        // scalar (stride 0); a causal diagonal tile advances by one int32 per row.
        configure_vec_iterator<VEC_V2_BASE>(0, 0, threshold_row_stride, 0);
    }
    configure_vec_iterator<VEC_V3_BASE>(0, 0, kVecBytes, 0);  // row state
    constexpr long instructions = MASK_SCORES ? kMaskedSoftmaxVecInstructions : kFullSoftmaxVecInstructions;
    begin_vec_block(instructions);

    // score_log2 = (QK) * scale * log2(e)
    __tx_vec_lsu_u2v<0, 0>(uram_addr(ctx_base + kCtxScore));
    __tx_vec_lsu_u2v<1, 1>(uram_addr(kUramScale));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();

    if constexpr (MASK_SCORES) {
        // s0 = lane <= per-row threshold.  pred bits are ordered <, =, >, so 0b110 is <=.
        __tx_vec_lsu_u2v<1, 1>(uram_addr(kUramLane));
        __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(ctx_base + kCtxThreshold));
        __tx_vec_vv_int32_cmp<16, 17, 0, 0b110>();
        __tx_vec_lsu_u2v<1, 1>(uram_addr(kUramFloor));
        __tx_vec_vv_mux2<0, 0, 16, 0>();  // valid -> score, invalid -> finite floor
    }
    __tx_vec_lsu_v2u<0, 0>(uram_addr(ctx_base + kCtxScore));

    emit_max_stage(32);
    emit_max_stage(16);
    emit_max_stage(8);
    emit_max_stage(4);
    emit_max_stage(2);
    emit_max_stage(1);

    // m_new = max(m_old, rowmax(score)).
    __tx_vec_lsu_u2v<1, 3>(uram_addr(ctx_base + kCtxM));
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(uram_addr(ctx_base + kCtxNewM));

    // P = 2^(score_log2 - m_new); the masked path explicitly clears invalid lanes after the SFU.
    __tx_vec_lsu_u2v<0, 0>(uram_addr(ctx_base + kCtxScore));
    __tx_vec_lsu_u2v<1, 3>(uram_addr(ctx_base + kCtxNewM));
    __tx_vec_vv_fp32_sub<0, 16, 0, 0>();
    // Clamp before SFU so no correctness argument depends on implementation-defined exp2(-inf).
    __tx_vec_lsu_u2v<1, 1>(uram_addr(kUramFloor));
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    if constexpr (MASK_SCORES) {
        // The finite mask floor can exponentiate to a nonzero value when the valid row maximum is
        // equally small.  Recreate the predicate and explicitly zero every invalid probability.
        __tx_vec_lsu_u2v<1, 1>(uram_addr(kUramLane));
        __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(ctx_base + kCtxThreshold));
        __tx_vec_vv_int32_cmp<16, 17, 0, 0b110>();
        __tx_vec_lsu_u2v<1, 1>(uram_addr(kUramZero));
        __tx_vec_vv_mux2<0, 0, 16, 0>();
    }
    __tx_vec_lsu_v2u<0, 0>(uram_addr(ctx_base + kCtxScore));

    emit_sum_stage(32);
    emit_sum_stage(16);
    emit_sum_stage(8);
    emit_sum_stage(4);
    emit_sum_stage(2);
    emit_sum_stage(1);
    __tx_vec_lsu_v2u<3, 0>(uram_addr(ctx_base + kCtxTileSum));

    // alpha = 2^(m_old - m_new).
    __tx_vec_lsu_u2v<0, 3>(uram_addr(ctx_base + kCtxM));
    __tx_vec_lsu_u2v<1, 3>(uram_addr(ctx_base + kCtxNewM));
    __tx_vec_vv_fp32_sub<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(uram_addr(kUramFloor));
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    __tx_vec_lsu_v2u<3, 0>(uram_addr(ctx_base + kCtxAlpha));

    // l_new = alpha * l_old + rowsum(P).
    __tx_vec_lsu_u2v<0, 3>(uram_addr(ctx_base + kCtxAlpha));
    __tx_vec_lsu_u2v<1, 3>(uram_addr(ctx_base + kCtxL));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 3>(uram_addr(ctx_base + kCtxTileSum));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(uram_addr(ctx_base + kCtxL));

    __tx_vec_lsu_u2v<0, 3>(uram_addr(ctx_base + kCtxNewM));
    __tx_vec_lsu_v2u<3, 0>(uram_addr(ctx_base + kCtxM));

    start_vec_block(instructions);
}

static inline void issue_online_softmax(long ctx_base, long rows, long threshold_row_stride, bool mask_scores) {
    if (mask_scores) {
        issue_online_softmax_impl<true>(ctx_base, rows, threshold_row_stride);
    } else {
        issue_online_softmax_impl<false>(ctx_base, rows, 0);
    }
}

constexpr long kUpdateOVecInstructions = 6;

static inline void issue_update_o(long ctx_base, long rows, long head_dim) {
    const long chunks = head_dim / kVecLanes;
    configure_vec_dims(rows, chunks);
    configure_vec_iterator<VEC_V0_BASE>(0, 0, head_dim * sizeof(float), kVecBytes);
    configure_vec_iterator<VEC_V1_BASE>(0, 0, kVecBytes, 0);
    configure_vec_iterator<VEC_V2_BASE>(0, 0, 0, 0);
    configure_vec_iterator<VEC_V3_BASE>(0, 0, 0, 0);
    begin_vec_block(kUpdateOVecInstructions);

    __tx_vec_lsu_u2v<0, 0>(uram_addr(ctx_base + kCtxO));
    __tx_vec_lsu_u2v<1, 0>(uram_addr(ctx_base + kCtxPV));
    __tx_vec_lsu_u2v<1, 1>(uram_addr(ctx_base + kCtxAlpha));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_vv_fp32_add<0, 17, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_addr(ctx_base + kCtxO));

    start_vec_block(kUpdateOVecInstructions);
}

constexpr long kNormalizeOVecInstructions = 5;

static inline void issue_normalize_o(long ctx_base, long rows, long head_dim) {
    const long chunks = head_dim / kVecLanes;
    configure_vec_dims(rows, chunks);
    configure_vec_iterator<VEC_V0_BASE>(0, 0, head_dim * sizeof(float), kVecBytes);
    configure_vec_iterator<VEC_V1_BASE>(0, 0, kVecBytes, 0);
    configure_vec_iterator<VEC_V2_BASE>(0, 0, 0, 0);
    configure_vec_iterator<VEC_V3_BASE>(0, 0, 0, 0);
    begin_vec_block(kNormalizeOVecInstructions);

    __tx_vec_lsu_u2v<0, 0>(uram_addr(ctx_base + kCtxO));
    __tx_vec_lsu_u2v<1, 1>(uram_addr(ctx_base + kCtxL));
    __tx_vec_v_fp32_sfu<1, 5, 16>();
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_addr(ctx_base + kCtxO));

    start_vec_block(kNormalizeOVecInstructions);
}

static inline void initialize_constants(float scale_log2e) {
    int32_t *   lane         = uram_ptr<int32_t>(kUramLane);
    float *     scale_vec    = uram_ptr<float>(kUramScale);
    float *     floor_vec    = uram_ptr<float>(kUramFloor);
    float *     zero_vec     = uram_ptr<float>(kUramZero);
    const float finite_floor = u32_as_float(0xff7fffffu);
    for (long i = 0; i < kVecLanes; ++i) {
        lane[i]      = static_cast<int32_t>(i);
        scale_vec[i] = scale_log2e;
        floor_vec[i] = finite_floor;
        zero_vec[i]  = 0.0f;
    }
}

// Initialize only the online-softmax state: m=-FLT_MAX, l=0, and O=0.  Q is loaded directly, while
// Score and PV are overwritten by MAT MM.  M and L share one row configuration; O is treated as a
// contiguous sequence of 256-byte VEC-sized chunks.
template <long GROUP> static inline void issue_initialize_online_state(long ctx_base, long rows, long head_dim) {
    const long chunks = head_dim / kVecLanes;
    configure_lsu_rows<GROUP>(rows, kVecBytes, 0, kVecBytes);
    __tx_lsu_dma_u2u<GROUP>(uram_addr(ctx_base + kCtxM), uram_addr(kUramFloor));
    __tx_lsu_dma_u2u<GROUP>(uram_addr(ctx_base + kCtxL), uram_addr(kUramZero));

    configure_lsu_rows<GROUP>(rows * chunks, kVecBytes, 0, kVecBytes);
    __tx_lsu_dma_u2u<GROUP>(uram_addr(ctx_base + kCtxO), uram_addr(kUramZero));
}

static inline bool prepare_softmax_mask(long   ctx_base,
                                        long   rows,
                                        long   real_cols,
                                        long   q_base,
                                        long   k_base,
                                        bool   causal,
                                        long * threshold_row_stride) {
    int32_t *  threshold     = uram_ptr<int32_t>(ctx_base + kCtxThreshold);
    const bool all_rows_full = !causal || q_base - k_base >= real_cols - 1;
    // A complete, fully valid tile takes the 63-instruction fast path and never reads a threshold.
    if (real_cols == kBc && all_rows_full) {
        *threshold_row_stride = 0;
        return false;
    }
    // A non-causal KV tail shares one scalar threshold across every row; only a causal diagonal
    // tile needs a compact per-row table.
    if (all_rows_full) {
        threshold[0]          = static_cast<int32_t>(real_cols - 1);
        *threshold_row_stride = 0;
        return true;
    }
    for (long r = 0; r < rows; ++r) {
        const long last_valid = min_long(real_cols - 1, q_base + r - k_base);
        threshold[r]          = static_cast<int32_t>(last_valid);
    }
    *threshold_row_stride = sizeof(int32_t);
    return true;
}

enum FlashAttentionError {
    FA_INVALID_ARGUMENT         = -1,
    FA_UNSUPPORTED_HEAD_DIM     = -2,
    FA_UNSUPPORTED_HEAD_MAPPING = -3,
    FA_SIZE_OVERFLOW            = -4,
    FA_UNALIGNED_V              = -5,
    FA_OUTPUT_ALIAS             = -6,
};

static inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t * result) {
    constexpr uint64_t max_u64 = ~uint64_t{ 0 };
    if (a != 0 && b > max_u64 / a) {
        return false;
    }
    *result = a * b;
    return true;
}

static inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t * result) {
    if (b > (~uint64_t{ 0 }) - a) {
        return false;
    }
    *result = a + b;
    return true;
}

static inline bool checked_strided_tensor_span(uint32_t   batches,
                                               uint32_t   heads,
                                               uint32_t   rows,
                                               uint64_t   row_bytes,
                                               uint64_t   batch_stride,
                                               uint64_t   head_stride,
                                               uint64_t   row_stride,
                                               uint64_t * result) {
    uint64_t batch_offset;
    uint64_t head_offset;
    uint64_t row_offset;
    uint64_t span;
    return checked_mul_u64(batches - 1U, batch_stride, &batch_offset) &&
           checked_mul_u64(heads - 1U, head_stride, &head_offset) &&
           checked_mul_u64(rows - 1U, row_stride, &row_offset) && checked_add_u64(batch_offset, head_offset, &span) &&
           checked_add_u64(span, row_offset, &span) && checked_add_u64(span, row_bytes, result);
}

static inline bool checked_tensor_bytes(uint32_t   d0,
                                        uint32_t   d1,
                                        uint32_t   d2,
                                        uint32_t   d3,
                                        uint64_t   element_bytes,
                                        uint64_t * result) {
    uint64_t value;
    return checked_mul_u64(d0, d1, &value) && checked_mul_u64(value, d2, &value) &&
           checked_mul_u64(value, d3, &value) && checked_mul_u64(value, element_bytes, result);
}

static inline bool valid_address_range(uint64_t base, uint64_t bytes) {
    return bytes <= (~uint64_t{ 0 }) - base;
}

// Call only after both ranges have passed valid_address_range().
static inline bool ranges_overlap(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
}

constexpr long kKvVtOffset = kUramVt0 - kUramK0;
static_assert(kUramVt1 - kUramK1 == kKvVtOffset, "Both KV pipeline buffers must use the same internal layout");

// A fast consumer can reach generation g+2 while another consumer still owns g in the same
// parity slot.  The generation tag prevents that consumer from mistaking the old READY tile for
// the new tile, matching the sequenced double buffer in kernel_bf16_matmul_os.cpp.
class NPUSequencedURAM : public NPUURAM {
  private:
    uint64_t generation = ~uint64_t{ 0 };

  public:
    explicit NPUSequencedURAM(uint64_t uram_base) : NPUURAM(uram_base) {}

    inline bool ready_for(uint64_t expected_generation) { return state == READY && generation == expected_generation; }

    inline int wait_ready_for(NPUTask * cur, NPUScheduler * sched, uint64_t expected_generation) {
        if (ready_for(expected_generation)) {
            return 0;
        }
        if (state == READY) {
            return wait_free(cur, sched);
        }
        return wait_ready(cur, sched);
    }

    inline void produce(NPUScheduler * sched, uint64_t produced_generation) {
        generation = produced_generation;
        NPUURAM::produce(sched);
    }
};

#define NPUWaitReadyFor(res, expected_generation)                          \
    do {                                                                   \
        while (!(res)->ready_for(expected_generation)) {                   \
            if ((res)->wait_ready_for(this, sched, expected_generation)) { \
                _state = &&CONCAT_LINE(_lb_);                              \
                return 0;                                                  \
                CONCAT_LINE(_lb_) :;                                       \
            }                                                              \
        }                                                                  \
    } while (0)

class FlashAttentionState {
  public:
    const uint64_t     q_base;
    const uint64_t     k_base;
    const uint64_t     v_base;
    const uint64_t     out_base;
    const uint64_t     q_row_stride;
    const uint64_t     q_head_stride;
    const uint64_t     q_batch_stride;
    const uint64_t     k_row_stride;
    const uint64_t     k_head_stride;
    const uint64_t     k_batch_stride;
    const uint64_t     v_row_stride;
    const uint64_t     v_head_stride;
    const uint64_t     v_batch_stride;
    const uint64_t     out_batch_bytes;
    const uint32_t     batch;
    const uint32_t     q_heads;
    const uint32_t     kv_heads;
    const uint32_t     q_len;
    const uint32_t     kv_len;
    const uint32_t     head_dim;
    const uint32_t     q_per_kv;
    const long         q_tiles;
    const long         q_groups;
    const long         kv_tiles;
    const uint32_t     causal_offset;
    const bool         causal;
    NPUSequencedURAM * kv_buffers;

    FlashAttentionState(uint64_t           q_base_arg,
                        uint64_t           k_base_arg,
                        uint64_t           v_base_arg,
                        uint64_t           out_base_arg,
                        uint32_t           batch_arg,
                        uint32_t           q_heads_arg,
                        uint32_t           kv_heads_arg,
                        uint32_t           q_len_arg,
                        uint32_t           kv_len_arg,
                        uint32_t           head_dim_arg,
                        uint32_t           causal_offset_arg,
                        bool               causal_arg,
                        uint64_t           q_row_stride_arg,
                        uint64_t           q_head_stride_arg,
                        uint64_t           q_batch_stride_arg,
                        uint64_t           k_row_stride_arg,
                        uint64_t           k_head_stride_arg,
                        uint64_t           k_batch_stride_arg,
                        uint64_t           v_row_stride_arg,
                        uint64_t           v_head_stride_arg,
                        uint64_t           v_batch_stride_arg,
                        NPUSequencedURAM * kv_buffers_arg) :
        q_base(q_base_arg),
        k_base(k_base_arg),
        v_base(v_base_arg),
        out_base(out_base_arg),
        q_row_stride(q_row_stride_arg),
        q_head_stride(q_head_stride_arg),
        q_batch_stride(q_batch_stride_arg),
        k_row_stride(k_row_stride_arg),
        k_head_stride(k_head_stride_arg),
        k_batch_stride(k_batch_stride_arg),
        v_row_stride(v_row_stride_arg),
        v_head_stride(v_head_stride_arg),
        v_batch_stride(v_batch_stride_arg),
        out_batch_bytes(static_cast<uint64_t>(q_heads_arg) * q_len_arg * head_dim_arg * sizeof(float)),
        batch(batch_arg),
        q_heads(q_heads_arg),
        kv_heads(kv_heads_arg),
        q_len(q_len_arg),
        kv_len(kv_len_arg),
        head_dim(head_dim_arg),
        q_per_kv(q_heads_arg / kv_heads_arg),
        q_tiles((static_cast<uint64_t>(q_len_arg) + kBr - 1) / kBr),
        q_groups((q_tiles + kConsumerCount - 1) / kConsumerCount),
        kv_tiles((static_cast<uint64_t>(kv_len_arg) + kBc - 1) / kBc),
        causal_offset(causal_offset_arg),
        causal(causal_arg),
        kv_buffers(kv_buffers_arg) {}

    inline uint32_t kv_head_for(uint32_t q_head) const { return q_head / q_per_kv; }

    inline uint64_t q_row_address(uint32_t b, uint32_t q_head, uint32_t row) const {
        return q_base + static_cast<uint64_t>(b) * q_batch_stride + static_cast<uint64_t>(q_head) * q_head_stride +
               static_cast<uint64_t>(row) * q_row_stride;
    }

    inline uint64_t k_row_address(uint32_t b, uint32_t q_head, uint32_t row) const {
        return k_base + static_cast<uint64_t>(b) * k_batch_stride +
               static_cast<uint64_t>(kv_head_for(q_head)) * k_head_stride + static_cast<uint64_t>(row) * k_row_stride;
    }

    inline uint64_t v_row_address(uint32_t b, uint32_t q_head, uint32_t row) const {
        return v_base + static_cast<uint64_t>(b) * v_batch_stride +
               static_cast<uint64_t>(kv_head_for(q_head)) * v_head_stride + static_cast<uint64_t>(row) * v_row_stride;
    }

    inline uint64_t out_row_address(uint32_t b, uint32_t q_head, uint32_t row) const {
        return out_base + static_cast<uint64_t>(b) * out_batch_bytes +
               (static_cast<uint64_t>(row) * q_heads + q_head) * head_dim * sizeof(float);
    }

    inline long q_tile_for(long q_group, long fixed_slice) const { return q_group * kConsumerCount + fixed_slice; }

    inline long rows_for_slice(long q_group, long fixed_slice) const {
        const long q_row = q_tile_for(q_group, fixed_slice) * kBr;
        if (q_row >= static_cast<long>(q_len)) {
            return 0;
        }
        return min_long(kBr, static_cast<long>(q_len) - q_row);
    }

    inline long slice_key_tile_count(long q_group, long fixed_slice) const {
        if (rows_for_slice(q_group, fixed_slice) == 0) {
            return 0;
        }
        const long q_tile     = q_tile_for(q_group, fixed_slice);
        const long last_query = min_long(static_cast<long>(q_len) - 1, q_tile * kBr + kBr - 1);
        return causal ? min_long(kv_tiles, (static_cast<long>(causal_offset) + last_query) / kBc + 1) : kv_tiles;
    }

    inline long producer_key_tile_count(long q_group) const {
        long count = 0;
        for (long slice = 0; slice < kConsumerCount; ++slice) {
            const long slice_count = slice_key_tile_count(q_group, slice);
            count                  = count < slice_count ? slice_count : count;
        }
        return count;
    }

    inline int users_for_key_tile(long q_group, long key_tile) const {
        int users = 0;
        for (long slice = 0; slice < kConsumerCount; ++slice) {
            users += slice_key_tile_count(q_group, slice) > key_tile;
        }
        return users;
    }
};

// Producer coroutine: pretranspose K/V tiles once for a super-group of up to three Q tiles.  A
// buffer remains READY until every consumer that uses that causal prefix has completed PV.
class FlashAttentionKvProducerTask : public NPUTask {
  private:
    FlashAttentionState * s;
    uint32_t              b             = 0;
    uint32_t              qh            = 0;
    long                  q_group       = 0;
    long                  ki            = 0;
    long                  producer_keys = 0;
    long                  k_row         = 0;
    long                  cols          = 0;
    int                   users         = 0;
    uint64_t              generation    = 0;
    uint64_t              k_tile        = 0;
    uint64_t              v_tile        = 0;
    NPUSequencedURAM *    kv            = nullptr;

  public:
    explicit FlashAttentionKvProducerTask(FlashAttentionState * state) : NPUTask(wrapper), s(state) {}

    int32_t execute(int * err) {
        (void) err;
        NPUBegin;
        for (b = 0; b < s->batch; ++b) {
            for (qh = 0; qh < s->q_heads; ++qh) {
                for (q_group = 0; q_group < s->q_groups; ++q_group) {
                    producer_keys = s->producer_key_tile_count(q_group);
                    for (ki = 0; ki < producer_keys; ++ki) {
                        kv    = &s->kv_buffers[generation & 1];
                        users = s->users_for_key_tile(q_group, ki);
                        do {
                            NPUWaitFree(kv);
                        } while (kv->allocate(sched, 1, users));

                        k_row  = ki * kBc;
                        cols   = min_long(kBc, static_cast<long>(s->kv_len) - k_row);
                        k_tile = s->k_row_address(b, qh, k_row);
                        v_tile = s->v_row_address(b, qh, k_row);

                        configure_lsu_rows<kLsuK>(cols, s->head_dim * sizeof(uint16_t), s->k_row_stride,
                                                  s->head_dim * sizeof(uint16_t));
                        __tx_lsu_dma_d2u<kLsuK>(uram_addr(kv->uram), k_tile);
                        load_v_transposed(v_tile, kv->uram + kKvVtOffset, cols, s->head_dim, s->v_row_stride);
                        NPUDMABar(0);

                        kv->produce(sched, generation);
                        ++generation;
                        NPUYield(0);
                    }
                }
            }
        }
        NPUFinish(-1);
    }

    static int32_t wrapper(NPUTask * task, int * err) {
        return static_cast<FlashAttentionKvProducerTask *>(task)->execute(err);
    }
};

// Three fixed consumers own disjoint URAM contexts and unit configuration groups.  Each handles
// one 64-row Q tile in a 192-row super-group, allowing LSU, MAT, VEC, and MACRO phases belonging
// to different Q tiles to cover one another's latency.  On the current RVTNG target this creates
// at most four outstanding LSU Produce markers and three on every other unit, below each 8-entry
// completion FIFO; FlexVec additionally has exactly three of its four block slots in flight.
template <long FIXED_SLICE> class FlashAttentionComputeTask : public NPUTask {
  private:
    static_assert(FIXED_SLICE >= 0 && FIXED_SLICE < kConsumerCount, "FlashAttention consumer slice is out of range");
    static constexpr long kCtxBase     = context_base(FIXED_SLICE);
    static constexpr long kLsuQGroup   = 2 + FIXED_SLICE * 2;
    static constexpr long kLsuOutGroup = kLsuQGroup + 1;
    static constexpr long kMatQKGroup  = FIXED_SLICE * 2;
    static constexpr long kMatPVGroup  = FIXED_SLICE * 2 + 1;
    static constexpr long kMacroGroup  = FIXED_SLICE;
    static_assert(kLsuOutGroup < 8 && kMatPVGroup < 8 && kMacroGroup < 8,
                  "FlashAttention unit group allocation exceeds the target TCSR file");

    FlashAttentionState * s;
    uint32_t              b                    = 0;
    uint32_t              qh                   = 0;
    long                  q_group              = 0;
    long                  qi                   = 0;
    long                  ki                   = 0;
    long                  q_row                = 0;
    long                  k_row                = 0;
    long                  rows                 = 0;
    long                  cols                 = 0;
    long                  producer_keys        = 0;
    long                  slice_keys           = 0;
    long                  threshold_row_stride = 0;
    bool                  mask_scores          = true;
    uint64_t              generation_base      = 0;
    uint64_t              expected_generation  = 0;
    NPUSequencedURAM *    kv                   = nullptr;

  public:
    explicit FlashAttentionComputeTask(FlashAttentionState * state) : NPUTask(wrapper), s(state) {}

    int32_t execute(int * err) {
        (void) err;
        NPUBegin;
        for (b = 0; b < s->batch; ++b) {
            for (qh = 0; qh < s->q_heads; ++qh) {
                for (q_group = 0; q_group < s->q_groups; ++q_group) {
                    producer_keys = s->producer_key_tile_count(q_group);
                    qi            = s->q_tile_for(q_group, FIXED_SLICE);
                    q_row         = qi * kBr;
                    rows          = s->rows_for_slice(q_group, FIXED_SLICE);
                    slice_keys    = s->slice_key_tile_count(q_group, FIXED_SLICE);
                    if (rows != 0) {
                        // Q needs no initialization: load it directly.  Initialize the independent
                        // online state on the output LSU group so both DMA streams can run in
                        // parallel; the existing barrier waits for both groups.
                        configure_lsu_rows<kLsuQGroup>(rows, s->head_dim * sizeof(uint16_t), s->q_row_stride,
                                                       s->head_dim * sizeof(uint16_t));
                        __tx_lsu_dma_d2u<kLsuQGroup>(uram_addr(kCtxBase + kCtxQ), s->q_row_address(b, qh, q_row));
                        issue_initialize_online_state<kLsuOutGroup>(kCtxBase, rows, s->head_dim);
                        NPUDMABar(0);

                        for (ki = 0; ki < slice_keys; ++ki) {
                            expected_generation = generation_base + static_cast<uint64_t>(ki);
                            kv                  = &s->kv_buffers[expected_generation & 1];
                            NPUWaitReadyFor(kv, expected_generation);

                            k_row = ki * kBc;
                            cols  = min_long(kBc, static_cast<long>(s->kv_len) - k_row);
                            issue_qk<kMatQKGroup>(kCtxBase, rows, cols, s->head_dim, kv->uram);
                            // Threshold preparation touches a disjoint URAM region and can run on
                            // the scalar core while QK is executing.
                            mask_scores = prepare_softmax_mask(kCtxBase, rows, cols, q_row + s->causal_offset, k_row,
                                                               s->causal, &threshold_row_stride);
                            NPUMatBar(0);

                            issue_online_softmax(kCtxBase, rows, threshold_row_stride, mask_scores);
                            NPUVecBar(0);

                            issue_p_to_bf16<kMacroGroup>(kCtxBase, rows);
                            NPUMACROBar(0);

                            issue_pv<kMatPVGroup>(kCtxBase, rows, s->head_dim, kv->uram + kKvVtOffset);
                            NPUMatBar(0);

                            // K/Vt are dead after PV.  Refill may overlap the independent O update.
                            kv->consume(sched);
                            issue_update_o(kCtxBase, rows, s->head_dim);
                            NPUVecBar(0);
                        }

                        issue_normalize_o(kCtxBase, rows, s->head_dim);
                        NPUVecBar(0);

                        configure_lsu_rows<kLsuOutGroup>(rows, s->head_dim * sizeof(float), s->head_dim * sizeof(float),
                                                         s->q_heads * s->head_dim * sizeof(float));
                        __tx_lsu_dma_u2d<kLsuOutGroup>(s->out_row_address(b, qh, q_row), uram_addr(kCtxBase + kCtxO));
                        NPUDMABar(0);
                    }
                    // All fixed consumers advance over the producer's full generation range,
                    // including causal suffixes they did not consume and an inactive tail slice.
                    generation_base += static_cast<uint64_t>(producer_keys);
                }
            }
        }
        NPUFinish(-1);
    }

    static int32_t wrapper(NPUTask * task, int * err) {
        return static_cast<FlashAttentionComputeTask<FIXED_SLICE> *>(task)->execute(err);
    }
};

}  // namespace

// Q/K/V rows are contiguous BF16.  The row, head, and batch strides describe their outer layout.
//   out : FP32 [batch, q_heads,  q_len,  head_dim]
// q_heads may be an integer multiple of kv_heads (grouped-query attention). flags bit 0 enables
// a causal mask and causal_offset aligns query row zero with its absolute KV-cache position.
// V must be 64-byte aligned, out must not overlap q/k/v, and Q/K/V values
// must be finite and small enough that FP32 MAT accumulation and scaled scores stay finite.
// scale_bits contains a finite nonnegative IEEE-754 FP32 bit pattern.  Returns 0 on success or
// FlashAttentionError.
__device__ long flash_attention_bf16_f32(void *   q,
                                         void *   k,
                                         void *   v,
                                         void *   out,
                                         uint32_t batch,
                                         uint32_t q_heads,
                                         uint32_t kv_heads,
                                         uint32_t q_len,
                                         uint32_t kv_len,
                                         uint32_t head_dim,
                                         uint32_t scale_bits,
                                         uint32_t causal_offset,
                                         uint32_t flags,
                                         uint64_t q_row_stride,
                                         uint64_t q_head_stride,
                                         uint64_t q_batch_stride,
                                         uint64_t k_row_stride,
                                         uint64_t k_head_stride,
                                         uint64_t k_batch_stride,
                                         uint64_t v_row_stride,
                                         uint64_t v_head_stride,
                                         uint64_t v_batch_stride) {
    const uint32_t scale_magnitude = scale_bits & 0x7fffffffu;
    if (!q || !k || !v || !out || batch == 0 || q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 ||
        (flags & ~kFlagCausal) != 0 || ((flags & kFlagCausal) == 0 && causal_offset != 0) ||
        ((flags & kFlagCausal) != 0 && static_cast<uint64_t>(causal_offset) + q_len > kv_len) ||
        (scale_bits & 0x7f800000u) == 0x7f800000u || ((scale_bits & 0x80000000u) != 0 && scale_magnitude != 0)) {
        return FA_INVALID_ARGUMENT;
    }
    if (head_dim == 0 || head_dim > kMaxHeadDim || (head_dim % kVecLanes) != 0) {
        return FA_UNSUPPORTED_HEAD_DIM;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(head_dim) * sizeof(uint16_t);
    if (q_row_stride < row_bytes || k_row_stride < row_bytes || v_row_stride < row_bytes ||
        ((q_row_stride | q_head_stride | q_batch_stride | k_row_stride | k_head_stride | k_batch_stride | v_row_stride |
          v_head_stride | v_batch_stride) &
         63U) != 0U) {
        return FA_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        return FA_UNSUPPORTED_HEAD_MAPPING;
    }
    const float scale_log2e = u32_as_float(scale_bits) * 1.4426950408889634f;
    if ((float_as_u32(scale_log2e) & 0x7f800000u) == 0x7f800000u) {
        return FA_INVALID_ARGUMENT;
    }
    // BF16 transpose uses 64-byte source transactions and therefore requires an aligned base.
    if ((reinterpret_cast<uint64_t>(v) & 63u) != 0) {
        return FA_UNALIGNED_V;
    }

    const uint64_t q_base_addr   = reinterpret_cast<uint64_t>(q);
    const uint64_t k_base_addr   = reinterpret_cast<uint64_t>(k);
    const uint64_t v_base_addr   = reinterpret_cast<uint64_t>(v);
    const uint64_t out_base_addr = reinterpret_cast<uint64_t>(out);
    uint64_t       q_span;
    uint64_t       k_span;
    uint64_t       v_span;
    uint64_t       out_bytes;
    uint64_t       traversal_generations;
    const uint64_t q_tiles  = (static_cast<uint64_t>(q_len) + kBr - 1) / kBr;
    const uint64_t q_groups = (q_tiles + kConsumerCount - 1) / kConsumerCount;
    const uint64_t kv_tiles = (static_cast<uint64_t>(kv_len) + kBc - 1) / kBc;
    if (!checked_strided_tensor_span(batch, q_heads, q_len, row_bytes, q_batch_stride, q_head_stride, q_row_stride,
                                     &q_span) ||
        !checked_strided_tensor_span(batch, kv_heads, kv_len, row_bytes, k_batch_stride, k_head_stride, k_row_stride,
                                     &k_span) ||
        !checked_strided_tensor_span(batch, kv_heads, kv_len, row_bytes, v_batch_stride, v_head_stride, v_row_stride,
                                     &v_span) ||
        !checked_tensor_bytes(batch, q_heads, q_len, head_dim, sizeof(float), &out_bytes) ||
        !checked_mul_u64(batch, q_heads, &traversal_generations) ||
        !checked_mul_u64(traversal_generations, q_groups, &traversal_generations) ||
        !checked_mul_u64(traversal_generations, kv_tiles, &traversal_generations) ||
        !valid_address_range(q_base_addr, q_span) || !valid_address_range(k_base_addr, k_span) ||
        !valid_address_range(v_base_addr, v_span) || !valid_address_range(out_base_addr, out_bytes)) {
        return FA_SIZE_OVERFLOW;
    }
    if (ranges_overlap(out_base_addr, out_bytes, q_base_addr, q_span) ||
        ranges_overlap(out_base_addr, out_bytes, k_base_addr, k_span) ||
        ranges_overlap(out_base_addr, out_bytes, v_base_addr, v_span)) {
        return FA_OUTPUT_ALIAS;
    }

    NPUSequencedURAM    kv_buffers[2] = { NPUSequencedURAM(kv_k_base(0)), NPUSequencedURAM(kv_k_base(1)) };
    FlashAttentionState state(q_base_addr, k_base_addr, v_base_addr, out_base_addr, batch, q_heads, kv_heads, q_len,
                              kv_len, head_dim, causal_offset, (flags & kFlagCausal) != 0, q_row_stride, q_head_stride,
                              q_batch_stride, k_row_stride, k_head_stride, k_batch_stride, v_row_stride, v_head_stride,
                              v_batch_stride, kv_buffers);
    initialize_constants(scale_log2e);
    FlashAttentionKvProducerTask producer(&state);
    FlashAttentionComputeTask<0> consumer0(&state);
    FlashAttentionComputeTask<1> consumer1(&state);
    FlashAttentionComputeTask<2> consumer2(&state);
    NPUTask *                    tasks[4] = { &producer, &consumer0, &consumer1, &consumer2 };
    NPUScheduler                 scheduler(4, tasks);
    for (NPUTask * task : tasks) {
        task->sched = &scheduler;
    }
    scheduler.scheduler();
    return 0;
}
