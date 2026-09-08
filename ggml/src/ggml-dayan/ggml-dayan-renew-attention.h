#pragma once

#include "ggml-dayan-activation.h"

#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#include <stdint.h>

namespace ggml_dayan_renew_attention_detail {

constexpr long kConsumerCount  = 3;
constexpr long kBr             = 64;
constexpr long kBc             = 128;
constexpr long kMaxHeadDim     = 256;
constexpr long kVecLanes       = 64;
constexpr long kVecBytes       = kVecLanes * sizeof(float);
constexpr long kLowpChunkBytes = kVecLanes * sizeof(uint16_t);
constexpr long kScalarStride   = 32;
constexpr long kMaskRowStride  = 64;
constexpr long kUramBytes      = 1024 * 1024;

constexpr uint32_t kFlagCausal = 1U;

constexpr long kQBytes     = kBr * kMaxHeadDim * sizeof(uint16_t);
constexpr long kKBytes     = kBc * kMaxHeadDim * sizeof(uint16_t);
constexpr long kVtBytes    = kMaxHeadDim * kBc * sizeof(uint16_t);
constexpr long kScoreBytes = kBr * kBc * sizeof(float);
constexpr long kPBytes     = kBr * kBc * sizeof(uint16_t);
constexpr long kPVBytes    = kBr * kMaxHeadDim * sizeof(float);
constexpr long kOBytes     = kBr * kMaxHeadDim * sizeof(float);
constexpr long kStateBytes = kBr * kScalarStride;
constexpr long kMaskBytes  = kBr * kMaskRowStride;

constexpr long kCtxQ         = 0x00000;
constexpr long kCtxScore     = 0x08000;
constexpr long kCtxP         = 0x10000;
constexpr long kCtxPV        = 0x14000;
constexpr long kCtxO         = 0x24000;
constexpr long kCtxM         = 0x34000;
constexpr long kCtxL         = 0x34800;
constexpr long kCtxAlpha     = 0x35000;
constexpr long kCtxMask      = 0x35800;
constexpr long kContextBytes = 0x36800;

constexpr long context_base(long consumer) {
    return consumer * kContextBytes;
}

constexpr long kUramK0  = context_base(kConsumerCount);
constexpr long kUramVt0 = kUramK0 + kKBytes;
constexpr long kUramK1  = kUramVt0 + kVtBytes;
constexpr long kUramVt1 = kUramK1 + kKBytes;
constexpr long kUramEnd = kUramVt1 + kVtBytes;

static_assert(kCtxScore == kCtxQ + kQBytes && kCtxP == kCtxScore + kScoreBytes && kCtxPV == kCtxP + kPBytes &&
                  kCtxO == kCtxPV + kPVBytes && kCtxM == kCtxO + kOBytes && kCtxL == kCtxM + kStateBytes &&
                  kCtxAlpha == kCtxL + kStateBytes && kCtxMask == kCtxAlpha + kStateBytes &&
                  kContextBytes == kCtxMask + kMaskBytes,
              "Renew Attention context layout is inconsistent");
static_assert(kUramEnd == 0xe3800 && kUramEnd <= kUramBytes, "Renew Attention URAM layout exceeds 1 MiB");

template <typename T> static inline T * uram_ptr(long byte_offset) {
    return reinterpret_cast<T *>(static_cast<char *>(dayan_uram_base()) + byte_offset);
}

static inline uint64_t uram_addr(long byte_offset) {
    return reinterpret_cast<uint64_t>(static_cast<char *>(dayan_uram_base()) + byte_offset);
}

static inline long min_long(long a, long b) {
    return a < b ? a : b;
}

static inline long clamp_long(long value, long low, long high) {
    return value < low ? low : (value > high ? high : value);
}

template <long Group> static inline void configure_lsu_rows(long row_bytes, long input_stride, long output_stride) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(row_bytes / 64 - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group> static inline void set_lsu_rows(long rows) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(rows - 1);
}

template <long Group> static inline void configure_v_transpose(long head_dim, uint64_t row_stride) {
    constexpr long side = 32;

    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(head_dim / side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(row_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(kBc * sizeof(uint16_t));
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(kBc * side * sizeof(uint16_t));
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group>
static inline void load_v_transposed(uint64_t v_dram, long vt_base, long rows, uint64_t row_stride) {
    constexpr long side   = 32;
    const long     panels = (rows + side - 1) / side;

    for (long panel = 0; panel < panels; ++panel) {
        const long panel_rows = min_long(side, rows - panel * side);
        __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(panel_rows - 1);
        __tx_lsu_dma_d2u<Group>(uram_addr(vt_base) + panel * side * sizeof(uint16_t),
                                v_dram + static_cast<uint64_t>(panel * side) * row_stride);
    }
}

template <typename Format, long Group>
static inline void configure_mat(long m, long n, long k, long a_stride, long b_stride, long c_stride) {
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_M>(m - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_N>(n - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_K>(k - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASTRIDE>(a_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSTRIDE>(b_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSTRIDE>(c_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DATATYPE>(Format::kMatDataType);
}

template <typename Format, long Group> static inline void issue_qk(long ctx_base, long cols, long k_base) {
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_N>(cols - 1);
    __tx_mat_mm<Group>(ctx_base + kCtxScore, ctx_base + kCtxQ, k_base);
}

template <long Group> static inline void issue_pv(long ctx_base, long vt_base) {
    __tx_mat_mm<Group>(ctx_base + kCtxPV, ctx_base + kCtxP, vt_base);
}

enum VecTcsr {
    VEC_CONTROL = 0,
    VEC_DIM0    = 2,
    VEC_DIM1    = 3,
    VEC_DIM2    = 4,
    VEC_DIM3    = 5,
    VEC_DIM4    = 6,
    VEC_DIM5    = 7,
    VEC_S0_BASE = 9,
    VEC_V0_BASE = 37,
    VEC_V1_BASE = 44,
    VEC_V2_BASE = 51,
    VEC_V3_BASE = 58,
};

static inline void initialize_vec_tcsr() {
    __tx_vec_tcsr_write<VEC_DIM0>(0);
    __tx_vec_tcsr_write<VEC_DIM1>(0);
    __tx_vec_tcsr_write<VEC_DIM4>(0);
    __tx_vec_tcsr_write<VEC_DIM5>(0);

    __tx_vec_tcsr_write<VEC_S0_BASE>(0);
    __tx_vec_tcsr_write<VEC_S0_BASE + 1>(0);
    __tx_vec_tcsr_write<VEC_S0_BASE + 2>(0);
    __tx_vec_tcsr_write<VEC_S0_BASE + 4>(0);
    __tx_vec_tcsr_write<VEC_S0_BASE + 5>(0);
    __tx_vec_tcsr_write<VEC_S0_BASE + 6>(0);

    __tx_vec_tcsr_write<VEC_V0_BASE>(0);
    __tx_vec_tcsr_write<VEC_V0_BASE + 1>(0);
    __tx_vec_tcsr_write<VEC_V0_BASE + 2>(0);
    __tx_vec_tcsr_write<VEC_V0_BASE + 4>(0);
    __tx_vec_tcsr_write<VEC_V0_BASE + 5>(0);
    __tx_vec_tcsr_write<VEC_V0_BASE + 6>(0);

    __tx_vec_tcsr_write<VEC_V1_BASE>(0);
    __tx_vec_tcsr_write<VEC_V1_BASE + 1>(0);
    __tx_vec_tcsr_write<VEC_V1_BASE + 2>(0);
    __tx_vec_tcsr_write<VEC_V1_BASE + 4>(0);
    __tx_vec_tcsr_write<VEC_V1_BASE + 5>(0);
    __tx_vec_tcsr_write<VEC_V1_BASE + 6>(0);

    __tx_vec_tcsr_write<VEC_V2_BASE>(0);
    __tx_vec_tcsr_write<VEC_V2_BASE + 1>(0);
    __tx_vec_tcsr_write<VEC_V2_BASE + 2>(0);
    __tx_vec_tcsr_write<VEC_V2_BASE + 3>(kScalarStride);
    __tx_vec_tcsr_write<VEC_V2_BASE + 4>(0);
    __tx_vec_tcsr_write<VEC_V2_BASE + 5>(0);
    __tx_vec_tcsr_write<VEC_V2_BASE + 6>(0);

    __tx_vec_tcsr_write<VEC_V3_BASE>(0);
    __tx_vec_tcsr_write<VEC_V3_BASE + 1>(0);
    __tx_vec_tcsr_write<VEC_V3_BASE + 2>(0);
    __tx_vec_tcsr_write<VEC_V3_BASE + 3>(kMaskRowStride);
    __tx_vec_tcsr_write<VEC_V3_BASE + 4>(0);
    __tx_vec_tcsr_write<VEC_V3_BASE + 5>(0);
    __tx_vec_tcsr_write<VEC_V3_BASE + 6>(0);
}

static inline void configure_vec_grid(long rows, long chunks = 1) {
    __tx_vec_tcsr_write<VEC_DIM2>(rows - 1);
    __tx_vec_tcsr_write<VEC_DIM3>(chunks - 1);
}

static inline void configure_softmax_grid(long rows) {
    configure_vec_grid(rows);
    __tx_vec_tcsr_write<VEC_V0_BASE + 3>(kBc * sizeof(float));
    __tx_vec_tcsr_write<VEC_V0_BASE + 4>(0);
    __tx_vec_tcsr_write<VEC_V1_BASE + 3>(kBc * sizeof(uint16_t));
    __tx_vec_tcsr_write<VEC_V1_BASE + 4>(0);
}

static inline void configure_output_grid(long rows, long head_dim) {
    configure_vec_grid(rows, head_dim / kVecLanes);
    __tx_vec_tcsr_write<VEC_V0_BASE + 3>(head_dim * sizeof(float));
    __tx_vec_tcsr_write<VEC_V0_BASE + 4>(kVecBytes);
    __tx_vec_tcsr_write<VEC_V1_BASE + 3>(head_dim * sizeof(float));
    __tx_vec_tcsr_write<VEC_V1_BASE + 4>(kVecBytes);
}

static inline long vec_control(long instructions) {
    const uint64_t n = static_cast<uint64_t>(instructions) & 0xffU;
    return static_cast<long>(n | (n << 8) | (n << 16) | (n << 24) | (n << 32));
}

static inline void begin_vec_block(long instructions) {
    long free_entries;
    do {
        __tx_vec_tcsr_read<VEC_CONTROL>(free_entries);
    } while (free_entries < instructions);
}

static inline void start_vec_block(long instructions) {
    // Keep CPU URAM stores before VEC submission.
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<VEC_CONTROL>(vec_control(instructions));
}

template <long Offset> static inline void emit_max_q0() {
    __tx_vec_shuffle_xor<1, Offset>(32);
    __tx_vec_vv_fp32_cmp<Offset, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, Offset, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(16);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(8);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(4);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(2);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(1);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
}

template <long Offset> static inline void emit_max_q1() {
    __tx_vec_shuffle_xor<0, 16 + Offset>(32);
    __tx_vec_vv_fp32_cmp<16 + Offset, 0, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16 + Offset, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(16);
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(8);
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(4);
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(2);
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(1);
    __tx_vec_vv_fp32_cmp<16, 0, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 0, 0>();
}

template <long Offset> static inline void emit_sum_q0() {
    __tx_vec_shuffle_xor<1, Offset>(32);
    __tx_vec_vv_fp32_add<0, Offset, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(16);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(8);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(4);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(2);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_shuffle_xor<1, 0>(1);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
}

template <long Offset> static inline void emit_sum_q1() {
    __tx_vec_shuffle_xor<0, 16 + Offset>(32);
    __tx_vec_vv_fp32_add<1, 16 + Offset, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(16);
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(8);
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(4);
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(2);
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(1);
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
}

constexpr long kFullSoftmaxInstructions   = 99;
constexpr long kMaskedSoftmaxInstructions = 107;

template <bool MaskScores, void (*ConvertP0)(), void (*ConvertP1)()>
static inline void issue_online_softmax(long ctx_base, uint32_t scale_bits) {
    constexpr long instructions    = MaskScores ? kMaskedSoftmaxInstructions : kFullSoftmaxInstructions;
    constexpr long old_m_offset    = MaskScores ? 5 : 4;
    constexpr long new_m_offset    = MaskScores ? 5 : 3;
    constexpr long floor_offset    = MaskScores ? 4 : 3;
    constexpr long stored_m_offset = MaskScores ? 8 : 6;

    begin_vec_block(instructions);
    __tx_vec_lsu_g2s<6>(1);
    __tx_vec_lsu_g2s<7>(0xffffffffU);

    if constexpr (MaskScores) {
        __tx_vec_lsu_u2s<4, 3>(uram_addr(ctx_base + kCtxMask));
        __tx_vec_lsu_u2s<5, 3>(uram_addr(ctx_base + kCtxMask + kScalarStride));
    }

    __tx_vec_lsu_u2v<0, 0>(uram_addr(ctx_base + kCtxScore));
    __tx_vec_lsu_g2v_broadcast32<1, 1>(scale_bits);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    if constexpr (MaskScores) {
        __tx_vec_lsu_g2v_broadcast32<1, 1>(0xff7fffffU);
        __tx_vec_vv_mux2<0, 16, 0, 4>();
    }
    __tx_vec_lsu_u2v<1, 0>(uram_addr(ctx_base + kCtxScore + kVecBytes));
    __tx_vec_vv_fp32_mul<1, 16, MaskScores ? 18 : 17, 0>();
    if constexpr (MaskScores) {
        __tx_vec_vv_mux2<1, 18, 16, 5>();
    }

    emit_max_q0<0>();
    emit_max_q1<6>();
    __tx_vec_vv_fp32_cmp<6, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 6, 16, 0>();

    __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(ctx_base + kCtxM));
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();

    __tx_vec_vv_fp32_sub<0, 14, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>(0xff7fffffU);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();

    __tx_vec_vv_fp32_sub<1, 30, 3, 0>();
    __tx_vec_vv_fp32_cmp<16, 17, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 17, 0>();
    __tx_vec_v_fp32_sfu<1, 16, 1>();

    if constexpr (MaskScores) {
        __tx_vec_lsu_g2v_broadcast32<0, 1>(0);
        __tx_vec_vv_mux2<0, 0, 1, 4>();
        __tx_vec_vv_mux2<1, 1, 16, 5>();
    }

    __tx_vec_vv_fp32_sub<0, 16 + old_m_offset, new_m_offset, 0>();
    __tx_vec_vv_fp32_cmp<0, 16 + floor_offset, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16 + floor_offset, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u_mask<0, 2, 6>(uram_addr(ctx_base + kCtxAlpha));
    __tx_vec_lsu_v2u_mask<stored_m_offset, 2, 6>(uram_addr(ctx_base + kCtxM));

    ConvertP0();
    __tx_vec_shuffle_compress<0, 0>(0);
    __tx_vec_lsu_v2u_mask<0, 1, 7>(uram_addr(ctx_base + kCtxP));
    ConvertP1();
    __tx_vec_shuffle_compress<1, 16>(0);
    __tx_vec_lsu_v2u_mask<16, 1, 7>(uram_addr(ctx_base + kCtxP + kLowpChunkBytes));

    emit_sum_q0<5>();
    emit_sum_q1<8>();
    __tx_vec_vv_fp32_add<0, 6, 16, 0>();

    __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(ctx_base + kCtxAlpha));
    __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(ctx_base + kCtxL));
    __tx_vec_vv_fp32_mul<1, 17, 16, 0>();
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_v2u_mask<0, 2, 6>(uram_addr(ctx_base + kCtxL));

    start_vec_block(instructions);
}

constexpr long kInitializeStateInstructions = 5;
constexpr long kInitializeOInstructions     = 2;

static inline void issue_initialize_state(long ctx_base, long rows, long head_dim) {
    configure_vec_grid(rows);
    begin_vec_block(kInitializeStateInstructions);
    __tx_vec_lsu_g2s<6>(1);
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0xff7fffffU);
    __tx_vec_lsu_v2u_mask<0, 2, 6>(uram_addr(ctx_base + kCtxM));
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
    __tx_vec_lsu_v2u_mask<0, 2, 6>(uram_addr(ctx_base + kCtxL));
    start_vec_block(kInitializeStateInstructions);

    configure_output_grid(rows, head_dim);
    begin_vec_block(kInitializeOInstructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
    __tx_vec_lsu_v2u<0, 0>(uram_addr(ctx_base + kCtxO));
    start_vec_block(kInitializeOInstructions);
}

constexpr long kInitializeVtInstructions = 2;

static inline void issue_initialize_vt(long head_dim) {
    configure_vec_grid(head_dim);
    __tx_vec_tcsr_write<VEC_V0_BASE + 3>(kBc * sizeof(uint16_t));
    __tx_vec_tcsr_write<VEC_V0_BASE + 4>(0);

    begin_vec_block(kInitializeVtInstructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
    __tx_vec_lsu_v2u<0, 0>(uram_addr(kUramVt0));
    start_vec_block(kInitializeVtInstructions);

    begin_vec_block(kInitializeVtInstructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
    __tx_vec_lsu_v2u<0, 0>(uram_addr(kUramVt1));
    start_vec_block(kInitializeVtInstructions);
}

constexpr long kUpdateOInstructions = 6;

static inline void issue_update_o(long ctx_base) {
    begin_vec_block(kUpdateOInstructions);
    __tx_vec_lsu_u2v<0, 0>(uram_addr(ctx_base + kCtxO));
    __tx_vec_lsu_u2v<1, 1>(uram_addr(ctx_base + kCtxPV));
    __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(ctx_base + kCtxAlpha));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_vv_fp32_add<0, 17, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_addr(ctx_base + kCtxO));
    start_vec_block(kUpdateOInstructions);
}

constexpr long kNormalizeOInstructions = 5;

static inline void issue_normalize_o(long ctx_base) {
    begin_vec_block(kNormalizeOInstructions);
    __tx_vec_lsu_u2v<0, 0>(uram_addr(ctx_base + kCtxO));
    __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_addr(ctx_base + kCtxL));
    __tx_vec_v_fp32_sfu<1, 16, 5>();
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_addr(ctx_base + kCtxO));
    start_vec_block(kNormalizeOInstructions);
}

static inline uint64_t low_bits(long count) {
    return count == 64 ? ~uint64_t{ 0 } : (count == 0 ? 0 : (uint64_t{ 1 } << count) - 1U);
}

static inline bool prepare_softmax_mask(long ctx_base,
                                        long rows,
                                        long real_cols,
                                        long q_base,
                                        long k_base,
                                        bool causal) {
    const bool all_rows_full = !causal || q_base - k_base >= real_cols - 1;
    if (real_cols == kBc && all_rows_full) {
        return false;
    }

    uint64_t * masks = uram_ptr<uint64_t>(ctx_base + kCtxMask);
    for (long row = 0; row < rows; ++row) {
        const long valid = clamp_long(causal ? q_base + row - k_base + 1 : real_cols, 0, real_cols);
        masks[row * kMaskRowStride / sizeof(uint64_t)] = low_bits(min_long(valid, kVecLanes));
        masks[(row * kMaskRowStride + kScalarStride) / sizeof(uint64_t)] =
            low_bits(clamp_long(valid - kVecLanes, 0, kVecLanes));
    }
    return true;
}

enum RenewAttentionError {
    RA_INVALID_ARGUMENT         = -1,
    RA_UNSUPPORTED_HEAD_DIM     = -2,
    RA_UNSUPPORTED_HEAD_MAPPING = -3,
    RA_SIZE_OVERFLOW            = -4,
    RA_UNALIGNED_V              = -5,
    RA_OUTPUT_ALIAS             = -6,
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

static inline bool ranges_overlap(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
}

constexpr long kKvVtOffset = kUramVt0 - kUramK0;
static_assert(kUramVt1 - kUramK1 == kKvVtOffset, "Renew Attention KV buffers have different layouts");

struct RenewAttentionBuffer {
    dayan_resource_t  resource;
    volatile uint64_t generation;
    long              k_offset;
};

struct RenewAttentionState {
    const uint64_t       q_base;
    const uint64_t       k_base;
    const uint64_t       v_base;
    const uint64_t       out_base;
    const uint64_t       q_row_stride;
    const uint64_t       q_head_stride;
    const uint64_t       q_batch_stride;
    const uint64_t       k_row_stride;
    const uint64_t       k_head_stride;
    const uint64_t       k_batch_stride;
    const uint64_t       v_row_stride;
    const uint64_t       v_head_stride;
    const uint64_t       v_batch_stride;
    const uint64_t       out_batch_bytes;
    const uint32_t       batch;
    const uint32_t       q_heads;
    const uint32_t       kv_heads;
    const uint32_t       q_len;
    const uint32_t       kv_len;
    const uint32_t       head_dim;
    const uint32_t       scale_log2e_bits;
    const uint32_t       task;
    const uint32_t       tasks;
    const uint32_t       q_per_kv;
    const long           q_tiles;
    const long           q_groups;
    const long           kv_tiles;
    const uint32_t       causal_offset;
    const bool           causal;
    RenewAttentionBuffer buffers[2];
    volatile int         status;
    volatile uint32_t    start;
    volatile uint32_t    abort;

    RenewAttentionState(uint64_t q_base_arg,
                        uint64_t k_base_arg,
                        uint64_t v_base_arg,
                        uint64_t out_base_arg,
                        uint32_t batch_arg,
                        uint32_t q_heads_arg,
                        uint32_t kv_heads_arg,
                        uint32_t q_len_arg,
                        uint32_t kv_len_arg,
                        uint32_t head_dim_arg,
                        uint32_t scale_log2e_bits_arg,
                        uint32_t causal_offset_arg,
                        bool     causal_arg,
                        uint64_t q_row_stride_arg,
                        uint64_t q_head_stride_arg,
                        uint64_t q_batch_stride_arg,
                        uint64_t k_row_stride_arg,
                        uint64_t k_head_stride_arg,
                        uint64_t k_batch_stride_arg,
                        uint64_t v_row_stride_arg,
                        uint64_t v_head_stride_arg,
                        uint64_t v_batch_stride_arg,
                        uint32_t task_arg,
                        uint32_t tasks_arg) :
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
        scale_log2e_bits(scale_log2e_bits_arg),
        task(task_arg),
        tasks(tasks_arg),
        q_per_kv(q_heads_arg / kv_heads_arg),
        q_tiles((static_cast<uint64_t>(q_len_arg) + kBr - 1) / kBr),
        q_groups((q_tiles + kConsumerCount - 1) / kConsumerCount),
        kv_tiles((static_cast<uint64_t>(kv_len_arg) + kBc - 1) / kBc),
        causal_offset(causal_offset_arg),
        causal(causal_arg),
        status(0),
        start(0),
        abort(0) {
        buffers[0].generation = ~uint64_t{ 0 };
        buffers[0].k_offset   = kUramK0;
        buffers[1].generation = ~uint64_t{ 0 };
        buffers[1].k_offset   = kUramK1;
    }

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
        return q_row >= static_cast<long>(q_len) ? 0 : min_long(kBr, static_cast<long>(q_len) - q_row);
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

static inline int remember_error(RenewAttentionState * state, int status) {
    if (status < 0 && state->status == 0) {
        state->status = status;
    }
    return status;
}

static inline int wait_for_generation(RenewAttentionBuffer * buffer, uint64_t expected_generation) {
    for (;;) {
        if (buffer->generation == expected_generation) {
            return dayan_resource_wait_ready(buffer->resource, (expected_generation / 2U) & 1U);
        }
        dayan_yield();
    }
}

template <typename Format, void (*ConvertP0)(), void (*ConvertP1)(), long FixedSlice>
static void attention_consumer(void * argument) {
    static_assert(FixedSlice >= 0 && FixedSlice < kConsumerCount, "Renew Attention consumer is out of range");
    constexpr long ctx_base = context_base(FixedSlice);
    constexpr long lsu_q    = 0;
    constexpr long lsu_out  = 1;
    constexpr long mat_qk   = 0;
    constexpr long mat_pv   = 1;

    RenewAttentionState * state           = static_cast<RenewAttentionState *>(argument);
    uint64_t              generation_base = 0;

    while (!state->start) {
        dayan_yield();
    }
    if (state->abort) {
        return;
    }

    configure_lsu_rows<lsu_q>(state->head_dim * sizeof(uint16_t), state->q_row_stride,
                              state->head_dim * sizeof(uint16_t));
    configure_lsu_rows<lsu_out>(state->head_dim * sizeof(float), state->head_dim * sizeof(float),
                                state->q_heads * state->head_dim * sizeof(float));
    initialize_vec_tcsr();

    const uint64_t work_items = static_cast<uint64_t>(state->batch) * state->q_heads;
    for (uint64_t work = state->task; work < work_items; work += state->tasks) {
        const uint32_t b  = static_cast<uint32_t>(work / state->q_heads);
        const uint32_t qh = static_cast<uint32_t>(work % state->q_heads);
        for (long q_group = 0; q_group < state->q_groups; ++q_group) {
            const long producer_keys = state->producer_key_tile_count(q_group);
            const long qi            = state->q_tile_for(q_group, FixedSlice);
            const long q_row         = qi * kBr;
            const long rows          = state->rows_for_slice(q_group, FixedSlice);
            const long slice_keys    = state->slice_key_tile_count(q_group, FixedSlice);

            if (rows != 0) {
                set_lsu_rows<lsu_q>(rows);
                __tx_lsu_dma_d2u<lsu_q>(uram_addr(ctx_base + kCtxQ), state->q_row_address(b, qh, q_row));
                issue_initialize_state(ctx_base, rows, state->head_dim);
                configure_mat<Format, mat_qk>(rows, kBc, state->head_dim, state->head_dim * sizeof(uint16_t),
                                              state->head_dim * sizeof(uint16_t), kBc * sizeof(float));
                configure_mat<Format, mat_pv>(rows, state->head_dim, kBc, kBc * sizeof(uint16_t),
                                              kBc * sizeof(uint16_t), state->head_dim * sizeof(float));
                dayan_wait_lsu();
                dayan_wait_vec();

                for (long ki = 0; ki < slice_keys; ++ki) {
                    const uint64_t         expected_generation = generation_base + static_cast<uint64_t>(ki);
                    RenewAttentionBuffer * buffer              = &state->buffers[expected_generation & 1U];
                    int                    status              = wait_for_generation(buffer, expected_generation);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }

                    const long k_row = ki * kBc;
                    const long cols  = min_long(kBc, static_cast<long>(state->kv_len) - k_row);
                    issue_qk<Format, mat_qk>(ctx_base, cols, buffer->k_offset);
                    const bool mask_scores =
                        prepare_softmax_mask(ctx_base, rows, cols, q_row + state->causal_offset, k_row, state->causal);
                    dayan_wait_mat();

                    configure_softmax_grid(rows);
                    if (mask_scores) {
                        issue_online_softmax<true, ConvertP0, ConvertP1>(ctx_base, state->scale_log2e_bits);
                    } else {
                        issue_online_softmax<false, ConvertP0, ConvertP1>(ctx_base, state->scale_log2e_bits);
                    }
                    dayan_wait_vec();

                    issue_pv<mat_pv>(ctx_base, buffer->k_offset + kKvVtOffset);
                    dayan_wait_mat();
                    status = dayan_resource_consume_mat(DAYAN_RESOURCE_MASK(buffer->resource));
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }

                    configure_output_grid(rows, state->head_dim);
                    issue_update_o(ctx_base);
                    dayan_wait_vec();
                }

                issue_normalize_o(ctx_base);
                dayan_wait_vec();
                set_lsu_rows<lsu_out>(rows);
                __tx_lsu_dma_u2d<lsu_out>(state->out_row_address(b, qh, q_row), uram_addr(ctx_base + kCtxO));
                dayan_wait_lsu();
            }
            generation_base += static_cast<uint64_t>(producer_keys);
        }
    }
    dayan_wait_mat();
}

static void attention_producer(void * argument) {
    constexpr long lsu_k  = 0;
    constexpr long lsu_vt = 1;

    RenewAttentionState * state = static_cast<RenewAttentionState *>(argument);

    while (!state->start) {
        dayan_yield();
    }
    if (state->abort) {
        return;
    }

    configure_lsu_rows<lsu_k>(state->head_dim * sizeof(uint16_t), state->k_row_stride,
                              state->head_dim * sizeof(uint16_t));
    configure_v_transpose<lsu_vt>(state->head_dim, state->v_row_stride);
    initialize_vec_tcsr();
    issue_initialize_vt(state->head_dim);
    dayan_wait_vec();

    int            status     = 0;
    uint64_t       generation = 0;
    const uint64_t work_items = static_cast<uint64_t>(state->batch) * state->q_heads;
    for (uint64_t work = state->task; work < work_items; work += state->tasks) {
        const uint32_t b  = static_cast<uint32_t>(work / state->q_heads);
        const uint32_t qh = static_cast<uint32_t>(work % state->q_heads);
        for (long q_group = 0; q_group < state->q_groups; ++q_group) {
            const long producer_keys = state->producer_key_tile_count(q_group);
            for (long ki = 0; ki < producer_keys; ++ki) {
                RenewAttentionBuffer * buffer = &state->buffers[generation & 1U];
                status                        = dayan_resource_wait_free(buffer->resource);
                if (status < 0) {
                    remember_error(state, status);
                    return;
                }
                buffer->generation = generation;
                const long k_row   = ki * kBc;
                const long cols    = min_long(kBc, static_cast<long>(state->kv_len) - k_row);
                set_lsu_rows<lsu_k>(cols);
                __tx_lsu_dma_d2u<lsu_k>(uram_addr(buffer->k_offset), state->k_row_address(b, qh, k_row));
                load_v_transposed<lsu_vt>(state->v_row_address(b, qh, k_row), buffer->k_offset + kKvVtOffset, cols,
                                          state->v_row_stride);
                status = dayan_resource_produce_lsu(buffer->resource, static_cast<uint32_t>(state->users_for_key_tile(q_group, ki)), 0);
                if (status < 0) {
                    remember_error(state, status);
                    return;
                }
                ++generation;
                dayan_yield();
            }
        }
    }
    dayan_wait_lsu();
}

template <typename Format, void (*ConvertP0)(), void (*ConvertP1)()>
static inline int run_pipeline(RenewAttentionState * state) {
    for (long index = 0; index < 2; ++index) {
        state->buffers[index].resource = DAYAN_LOCAL_RESOURCE(index);
    }

    dayan_thread_resources_t resources;
    dayan_thread_resources_init(&resources);
    resources.local_resource_count = 2;
    resources.lsu_tcsr_count = 2;
    resources.vec_tcsr_count = 1;
    int status               = dayan_create_thread("renew-producer", attention_producer, state, &resources);
    resources.mat_tcsr_count = 2;
    if (status >= 0) {
        status =
            dayan_create_thread("renew-cons-0", attention_consumer<Format, ConvertP0, ConvertP1, 0>, state, &resources);
    }
    if (status >= 0) {
        status =
            dayan_create_thread("renew-cons-1", attention_consumer<Format, ConvertP0, ConvertP1, 1>, state, &resources);
    }
    if (status >= 0) {
        status =
            dayan_create_thread("renew-cons-2", attention_consumer<Format, ConvertP0, ConvertP1, 2>, state, &resources);
    }
    if (status < 0) {
        state->status = status;
        state->abort  = 1;
        state->start  = 1;
        (void) dayan_thread_join_all();
        return status;
    }

    state->start = 1;
    status       = dayan_thread_join_all();
    if (status < 0 && state->status == 0) {
        state->status = status;
    }
    return state->status;
}

template <typename Format, void (*ConvertP0)(), void (*ConvertP1)()>
static __device__ long renew_attention_lowp_f32(void *   q,
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
    const uint32_t task            = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks           = dayan_task_dim(DAYAN_TASK_X);
    const uint32_t scale_magnitude = scale_bits & 0x7fffffffU;
    if (!q || !k || !v || !out || batch == 0 || q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 ||
        (flags & ~kFlagCausal) != 0 || ((flags & kFlagCausal) == 0 && causal_offset != 0) ||
        ((flags & kFlagCausal) != 0 && static_cast<uint64_t>(causal_offset) + q_len > kv_len) ||
        (scale_bits & 0x7f800000U) == 0x7f800000U || ((scale_bits & 0x80000000U) != 0 && scale_magnitude != 0) ||
        dayan_task_mode() != 1U || tasks == 0U || tasks > kDayanNativeMaxTasks || task >= tasks ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        return RA_INVALID_ARGUMENT;
    }
    if (head_dim == 0 || head_dim > kMaxHeadDim || (head_dim % kVecLanes) != 0) {
        return RA_UNSUPPORTED_HEAD_DIM;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(head_dim) * sizeof(uint16_t);
    if (q_row_stride < row_bytes || k_row_stride < row_bytes || v_row_stride < row_bytes ||
        ((q_row_stride | q_head_stride | q_batch_stride | k_row_stride | k_head_stride | k_batch_stride | v_row_stride |
          v_head_stride | v_batch_stride) &
         63U) != 0U) {
        return RA_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        return RA_UNSUPPORTED_HEAD_MAPPING;
    }

    const float scale_log2e = u32_as_float(scale_bits) * 1.4426950408889634f;
    if ((float_as_u32(scale_log2e) & 0x7f800000U) == 0x7f800000U) {
        return RA_INVALID_ARGUMENT;
    }
    if ((reinterpret_cast<uint64_t>(v) & 63U) != 0) {
        return RA_UNALIGNED_V;
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
        return RA_SIZE_OVERFLOW;
    }
    if (ranges_overlap(out_base_addr, out_bytes, q_base_addr, q_span) ||
        ranges_overlap(out_base_addr, out_bytes, k_base_addr, k_span) ||
        ranges_overlap(out_base_addr, out_bytes, v_base_addr, v_span)) {
        return RA_OUTPUT_ALIAS;
    }
    if (!dayan_uram_base() || dayan_uram_size() < kUramBytes) {
        return RA_INVALID_ARGUMENT;
    }

    RenewAttentionState state(q_base_addr, k_base_addr, v_base_addr, out_base_addr, batch, q_heads, kv_heads, q_len,
                              kv_len, head_dim, float_as_u32(scale_log2e), causal_offset, (flags & kFlagCausal) != 0,
                              q_row_stride, q_head_stride, q_batch_stride, k_row_stride, k_head_stride, k_batch_stride,
                              v_row_stride, v_head_stride, v_batch_stride, task, tasks);
    return run_pipeline<Format, ConvertP0, ConvertP1>(&state);
}

}  // namespace ggml_dayan_renew_attention_detail
