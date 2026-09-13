#pragma once

#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#include <stdint.h>

namespace ggml_dayan_gdn_detail {

constexpr uint32_t kHeadDim     = 128;
constexpr long     kLanes       = 64;
constexpr long     kVecBytes    = kLanes * sizeof(float);
constexpr long     kScalarStride = 32;
constexpr long     kRowsPerGroup = kLanes;
constexpr long     kRowGroups    = kHeadDim / kRowsPerGroup;
constexpr long     kRowsPerThread = 8;
constexpr long     kMaskGroupBytes = kRowsPerGroup * kScalarStride;

constexpr long kUramState       = 0x00000;
constexpr long kUramQ           = 0x10000;
constexpr long kUramK           = 0x10200;
constexpr long kUramV           = 0x10400;
constexpr long kUramG           = 0x10600;
constexpr long kUramGate        = 0x10800;
constexpr long kUramGatedK      = 0x10a00;
constexpr long kUramMaskTable   = 0x10c00;
constexpr long kUramOutputPacked = kUramMaskTable + kRowGroups * kMaskGroupBytes;
constexpr long kUramBeta        = kUramOutputPacked + kMaskGroupBytes + kVecBytes;
constexpr long kUramLog2E       = kUramBeta + 64;
constexpr long kUramEnd         = kUramLog2E + 64;

static_assert(kUramEnd <= DAYAN_GATED_DELTA_NET_TASK_URAM_BYTES, "Gated Delta Net URAM layout is too large");

enum VecTcsr {
    kVecControl = 0,
    kVecSync    = 1,
    kVecDim0    = 2,
    kVecDim1    = 3,
    kVecDim2    = 4,
    kVecDim3    = 5,
    kVecDim4    = 6,
    kVecDim5    = 7,
    kVecS0Base  = 9,
    kVecV0Base  = 37,
    kVecV1Base  = 44,
    kVecV2Base  = 51,
    kVecV3Base  = 58,
};

static inline uint64_t uram_address(long offset) {
    return reinterpret_cast<uint64_t>(static_cast<char *>(dayan_uram_base()) + offset);
}

template <typename T> static inline T * uram_pointer(long offset) {
    return reinterpret_cast<T *>(uram_address(offset));
}

template <long Base>
static inline void configure_vec_iterator(long dim0_stride, long dim1_stride, long dim2_stride, long dim3_stride) {
    __tx_vec_tcsr_write<Base>(0);
    __tx_vec_tcsr_write<Base + 1>(dim0_stride);
    __tx_vec_tcsr_write<Base + 2>(dim1_stride);
    __tx_vec_tcsr_write<Base + 3>(dim2_stride);
    __tx_vec_tcsr_write<Base + 4>(dim3_stride);
    __tx_vec_tcsr_write<Base + 5>(0);
    __tx_vec_tcsr_write<Base + 6>(0);
}

static inline void configure_vec_recurrence() {
    // Each outer context updates eight state rows and packs eight outputs.
    __tx_vec_tcsr_write<kVecDim0>(0);
    __tx_vec_tcsr_write<kVecDim1>(kRowsPerThread - 1);
    __tx_vec_tcsr_write<kVecDim2>(kRowsPerGroup / kRowsPerThread - 1);
    __tx_vec_tcsr_write<kVecDim3>(kRowGroups - 1);
    __tx_vec_tcsr_write<kVecDim4>(0);
    __tx_vec_tcsr_write<kVecDim5>(0);

    configure_vec_iterator<kVecS0Base>(0, 1, 0, 0);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, kHeadDim * sizeof(float),
                                       kRowsPerThread * kHeadDim * sizeof(float),
                                       kRowsPerGroup * kHeadDim * sizeof(float));
    configure_vec_iterator<kVecV2Base>(0, 0, kRowsPerThread * sizeof(float), kVecBytes);
    configure_vec_iterator<kVecV3Base>(0, kScalarStride, 0, 0);
}

static inline long vec_control(long prologue0, long prologue1, long loop, long epilogue1, long epilogue0) {
    return static_cast<long>((static_cast<uint64_t>(prologue0) & 0xffU) |
                             ((static_cast<uint64_t>(prologue1) & 0xffU) << 8) |
                             ((static_cast<uint64_t>(loop) & 0xffU) << 16) |
                             ((static_cast<uint64_t>(epilogue1) & 0xffU) << 24) |
                             ((static_cast<uint64_t>(epilogue0) & 0xffU) << 32));
}

static inline void begin_vec_block(long instructions) {
    long free_entries;
    do {
        __tx_vec_tcsr_read<kVecControl>(free_entries);
    } while (free_entries < instructions);
}

static inline void start_vec_block(long prologue0, long prologue1, long loop, long epilogue1, long epilogue0) {
    // Keep CPU URAM stores before VEC submission.
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecSync>(0);
    __tx_vec_tcsr_write<kVecControl>(vec_control(prologue0, prologue1, loop, epilogue1, epilogue0));
}

template <long Group>
static inline void configure_lsu_rows(long rows, long row_bytes, long input_stride, long output_stride) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(row_bytes / 64 - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(rows - 1);
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

static inline void emit_sum_64() {
    __tx_vec_shuffle_xor<0, 0>(32);
    __tx_vec_vv_fp32_add<0, 0, 1, 0>();
    __tx_vec_shuffle_xor<0, 0>(16);
    __tx_vec_vv_fp32_add<0, 0, 1, 0>();
    __tx_vec_shuffle_xor<0, 0>(8);
    __tx_vec_vv_fp32_add<0, 0, 1, 0>();
    __tx_vec_shuffle_xor<0, 0>(4);
    __tx_vec_vv_fp32_add<0, 0, 1, 0>();
    __tx_vec_shuffle_xor<0, 0>(2);
    __tx_vec_vv_fp32_add<0, 0, 1, 0>();
    __tx_vec_shuffle_xor<0, 0>(1);
    __tx_vec_vv_fp32_add<0, 0, 1, 0>();
}

static inline void emit_dot_half(uint64_t input, uint64_t state) {
    __tx_vec_lsu_u2v<0, 0>(input);
    __tx_vec_lsu_u2v<0, 1>(state);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    emit_sum_64();
}

template <long Delta>
static inline void emit_update_half(uint64_t state, uint64_t key, uint64_t query) {
    __tx_vec_lsu_u2v<0, 1>(state);
    __tx_vec_lsu_u2v<0, 0>(key);
    __tx_vec_vv_fp32_mul<0, Delta, 0, 0>();
    __tx_vec_vv_fp32_add<0, 0, 2, 0>();
    __tx_vec_lsu_v2u<0, 1>(state);
    __tx_vec_lsu_u2v<0, 0>(query);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    emit_sum_64();
}

static inline void configure_vec_outer(long rows, long halves = 1) {
    __tx_vec_tcsr_write<kVecDim0>(0);
    __tx_vec_tcsr_write<kVecDim1>(0);
    __tx_vec_tcsr_write<kVecDim2>(rows - 1);
    __tx_vec_tcsr_write<kVecDim3>(halves - 1);
    __tx_vec_tcsr_write<kVecDim4>(0);
    __tx_vec_tcsr_write<kVecDim5>(0);
}

static inline void issue_scalar_gate(uint64_t input) {
    constexpr long instructions = 5;
    configure_vec_outer(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>(input);
    __tx_vec_lsu_u2v_broadcast32<1, 1>(uram_address(kUramLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 3>(uram_address(kUramGate));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
    dayan_wait_vec();
}

static inline void issue_vector_gate(uint64_t input) {
    constexpr long instructions = 5;
    configure_vec_outer(kRowGroups);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    __tx_vec_lsu_u2v_broadcast32<1, 1>(uram_address(kUramLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 3>(uram_address(kUramGate));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
    dayan_wait_vec();
}

template <long GateHalfStride>
static inline void issue_scale_state() {
    constexpr long instructions = 4;
    configure_vec_outer(kHeadDim, kRowGroups);
    configure_vec_iterator<kVecV0Base>(0, 0, kHeadDim * sizeof(float), kVecBytes);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, GateHalfStride);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kHeadDim * sizeof(float), kVecBytes);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_address(kUramState));
    __tx_vec_lsu_u2v<1, 1>(uram_address(kUramGate));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>(uram_address(kUramState));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
    dayan_wait_vec();
}

static inline void emit_recurrence_row(uint32_t scale_bits) {
    emit_dot_half(uram_address(kUramK), uram_address(kUramState));
    __tx_vec_shuffle_xor<1, 0>(0);
    emit_dot_half(uram_address(kUramK + kVecBytes), uram_address(kUramState + kVecBytes));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_s_iter_gets<0, 0, 0, 0>();
    __tx_vec_shuffle_broadcast_dyn<0, 17, 0>();
    __tx_vec_vv_fp32_sub<0, 0, 1, 0>();
    __tx_vec_lsu_u2v_broadcast32<0, 0>(uram_address(kUramBeta));
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_shuffle_xor<1, 0>(0);

    emit_update_half<16>(uram_address(kUramState), uram_address(kUramK), uram_address(kUramQ));
    __tx_vec_shuffle_xor<1, 0>(0);
    emit_update_half<17>(uram_address(kUramState + kVecBytes), uram_address(kUramK + kVecBytes),
                         uram_address(kUramQ + kVecBytes));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 0>(scale_bits);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_lsu_u2s<0, 3>(uram_address(kUramMaskTable));
    __tx_vec_vv_mux2<1, 20, 0, 0>();
    __tx_vec_shuffle_xor<1, 20>(0);
}

static inline void issue_recurrence(uint32_t scale_bits) {
    constexpr long prologue0 = 0;
    constexpr long group_prologue = 2;
    constexpr long row_instructions = 83;
    constexpr long group_epilogue = 2;
    const long prologue1 = prologue0 + group_prologue;
    const long loop = prologue1 + row_instructions;
    const long epilogue1 = loop + group_epilogue;

    begin_vec_block(epilogue1);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    __tx_vec_lsu_u2v<1, 2>(uram_address(kUramV));
    emit_recurrence_row(scale_bits);
    __tx_vec_lsu_g2s<0>((1U << kRowsPerThread) - 1);
    __tx_vec_lsu_v2u_mask<17, 2, 0>(uram_address(kUramOutputPacked));
    start_vec_block(prologue0, prologue1, loop, epilogue1, epilogue1);
}

template <long Group> static inline void configure_lsu_scalar() {
    configure_lsu_rows<Group>(1, 64, 64, 64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(sizeof(float) - 1);
}

static inline const float * gdn_row(const float *  base,
                                    const uint64_t nb[4],
                                    uint32_t       head,
                                    uint32_t       token,
                                    uint32_t       sequence) {
    return reinterpret_cast<const float *>(reinterpret_cast<const char *>(base) + (uint64_t) head * nb[1] +
                                           (uint64_t) token * nb[2] + (uint64_t) sequence * nb[3]);
}

constexpr long kChunkTokens = 64;
constexpr long kHeadStride = kHeadDim * sizeof(float) + 64;
constexpr long kChunkStride = kChunkTokens * sizeof(float) + 64;
constexpr long kChunkState = 0;
constexpr long kChunkQ = kChunkState + kHeadDim * kHeadStride;
constexpr long kChunkK = kChunkQ + kChunkTokens * kHeadStride;
constexpr long kChunkV = kChunkK + kChunkTokens * kHeadStride;
constexpr long kChunkR = kChunkV + kChunkTokens * kHeadStride;
constexpr long kChunkO = kChunkR + kHeadDim * kChunkStride;
constexpr long kChunkKK = kChunkO + kHeadDim * kChunkStride;
constexpr long kChunkKQ = kChunkKK + kChunkTokens * kChunkStride;
constexpr long kChunkTemp = kChunkKQ + kChunkTokens * kChunkStride;
constexpr long kChunkG = kChunkTemp + kHeadDim * kChunkStride;
constexpr long kChunkBeta = kChunkG + kVecBytes;
constexpr long kChunkGCum = kChunkBeta + kVecBytes;
constexpr long kChunkExpG = kChunkGCum + kVecBytes;
constexpr long kChunkTailDecay = kChunkExpG + kVecBytes;
constexpr long kChunkEndDecay = kChunkTailDecay + kVecBytes;
constexpr long kChunkLower = kChunkEndDecay + kVecBytes;
constexpr long kChunkDiag = kChunkLower + kChunkTokens * kScalarStride;
constexpr long kChunkEnd = kChunkDiag + kChunkTokens * kScalarStride;
static_assert(kChunkEnd <= DAYAN_GATED_DELTA_NET_TASK_URAM_BYTES, "GDN chunk exceeds URAM budget");

static inline void configure_chunk_mat(long m, long n, long k, long a_stride, long b_stride, long c_stride) {
    __tx_mat_tcsr_write<MAT_M>(m - 1);
    __tx_mat_tcsr_write<MAT_N>(n - 1);
    __tx_mat_tcsr_write<MAT_K>(k - 1);
    __tx_mat_tcsr_write<MAT_ASTRIDE>(a_stride);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(b_stride);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(c_stride);
}

static inline void issue_chunk_prefix(long rows) {
    constexpr long instructions = 38;
    configure_vec_outer(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkG));
    for (long shift = 1; shift < kChunkTokens; shift *= 2) {
        __tx_vec_lsu_g2s<0>((long) (~uint64_t{0} << shift));
        __tx_vec_shuffle_ror<0, 0>(kChunkTokens - shift);
        __tx_vec_vv_fp32_add<0, 0, 1, 0>();
        __tx_vec_vv_mux2<0, 2, 0, 0>();
    }
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkGCum));
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0x3fb8aa3bU);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkExpG));
    __tx_vec_shuffle_broadcast<0, 0>(rows - 1);
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkEndDecay));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkGCum));
    __tx_vec_shuffle_broadcast<0, 0>(rows - 1);
    __tx_vec_vv_fp32_sub<0, 0, 1, 0>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkTailDecay));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
}

static inline void issue_chunk_coefficients(long rows, uint32_t scale_bits) {
    constexpr long instructions = 25;
    configure_vec_outer(rows);
    configure_vec_iterator<kVecS0Base>(0, 0, 1, 0);
    configure_vec_iterator<kVecV0Base>(0, 0, kChunkStride, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kScalarStride, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<1, 1>(uram_address(kChunkGCum));
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    __tx_vec_s_iter_gets<0, 0, 0, 0>();
    __tx_vec_shuffle_broadcast_dyn<0, 17, 0>();
    __tx_vec_vv_fp32_sub<0, 17, 0, 0>();
    // Mask unused entries before exp to avoid overflow.
    __tx_vec_lsu_u2s<0, 3>(uram_address(kChunkDiag));
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0x3fb8aa3bU);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_shuffle_xor<1, 0>(0);
    __tx_vec_lsu_u2s<0, 3>(uram_address(kChunkLower));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkKK));
    __tx_vec_lsu_u2v<0, 1>(uram_address(kChunkBeta));
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_vv_mux2<0, 17, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkKK));
    __tx_vec_lsu_u2s<0, 3>(uram_address(kChunkDiag));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkKQ));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 0>(scale_bits);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_vv_mux2<0, 17, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkKQ));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
}

static inline void issue_chunk_residual(uint64_t valid_mask, uint32_t scale_bits) {
    constexpr long instructions = 17;
    configure_vec_outer(kHeadDim);
    configure_vec_iterator<kVecV0Base>(0, 0, kChunkStride, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<1, 1>(uram_address(kChunkExpG));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkR));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkTemp));
    __tx_vec_vv_fp32_sub<0, 0, 1, 0>();
    __tx_vec_lsu_u2v<1, 1>(uram_address(kChunkBeta));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    __tx_vec_lsu_g2s<0>((long) valid_mask);
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkR));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkO));
    __tx_vec_vv_fp32_mul<0, 0, 18, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 0>(scale_bits);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkO));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
}

static inline void issue_chunk_solve(long rows) {
    if (rows == 1) {
        return;
    }
    configure_vec_outer(kHeadDim);
    __tx_vec_tcsr_write<kVecDim1>(rows - 2);
    configure_vec_iterator<kVecS0Base>(0, 1, 0, 0);
    configure_vec_iterator<kVecV0Base>(0, kChunkStride, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, kChunkStride, 0);
    begin_vec_block(7);
    // Eliminate one column at a time; every value row retains all 64 RHS lanes.
    __tx_vec_lsu_u2v<1, 1>(uram_address(kChunkR));
    __tx_vec_s_iter_gets<0, 0, 0, 0>();
    __tx_vec_shuffle_broadcast_dyn<0, 16, 0>();
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkKK));
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_vv_fp32_sub<1, 16, 0, 0>();
    __tx_vec_lsu_v2u<16, 1>(uram_address(kChunkR));
    start_vec_block(1, 1, 6, 6, 7);
}

static inline void issue_chunk_decay(long rows) {
    configure_vec_outer(kRowGroups, rows);
    configure_vec_iterator<kVecS0Base>(0, 0, 0, 1);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, kHeadStride);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    begin_vec_block(6);
    __tx_vec_lsu_u2v<1, 1>(uram_address(kChunkTailDecay));
    __tx_vec_s_iter_gets<0, 0, 0, 0>();
    __tx_vec_shuffle_broadcast_dyn<0, 16, 0>();
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkK));
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkK));
    start_vec_block(6, 6, 6, 6, 6);
    dayan_wait_vec();

    configure_vec_outer(kRowGroups, kHeadDim);
    begin_vec_block(4);
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkState));
    __tx_vec_lsu_u2v_broadcast32<1, 1>(uram_address(kChunkEndDecay));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkState));
    start_vec_block(4, 4, 4, 4, 4);
}

static inline void run_chunked_head(const DayanGatedDeltaNetConfig * config, const float * q, const float * k,
                                    const float * v, const float * g, const float * beta, const float * state,
                                    float * output, uint32_t head, uint32_t sequence) {
    const uint64_t state_index = ((uint64_t) sequence * config->heads + head) * kHeadDim * kHeadDim;
    const uint64_t attn_count = (uint64_t) config->sequences * config->tokens * config->heads * kHeadDim;
    const uint32_t q_sequence = sequence / (config->sequences / config->q_sequences);
    const uint32_t k_sequence = sequence / (config->sequences / config->k_sequences);
    long checksum = 0;
    configure_lsu_rows<0>(kHeadDim, kHeadDim * sizeof(float), kHeadDim * sizeof(float), kHeadStride);
    __tx_lsu_dma_d2u<0>(uram_address(kChunkState), (uint64_t) (state + state_index));
    dayan_wait_lsu();
    // Padded linear rows keep VEC access compatible and give MAT a 1x Bank peak.
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_TF32, BaseNPUInstBuilder::DT_MATIN_TF32));

    for (uint64_t token = 0; token < config->tokens; token += kChunkTokens) {
        const long rows = config->tokens - token < kChunkTokens ? config->tokens - token : kChunkTokens;
        const uint64_t valid_mask = rows == kChunkTokens ? ~uint64_t{0} : (uint64_t{1} << rows) - 1;
        for (long j = 0; j < rows; ++j) {
            *uram_pointer<uint64_t>(kChunkLower + j * kScalarStride) = j == 63 ? 0 : valid_mask & (~uint64_t{0} << (j + 1));
            *uram_pointer<uint64_t>(kChunkDiag + j * kScalarStride) = valid_mask & (~uint64_t{0} << j);
        }
        if (rows != kChunkTokens) {
            configure_vec_outer(3 * kChunkTokens * kHeadStride / kVecBytes);
            configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
            begin_vec_block(2);
            __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
            __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkQ));
            start_vec_block(2, 2, 2, 2, 2);
            dayan_wait_vec();
        }
        configure_vec_outer(2);
        configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
        begin_vec_block(2);
        __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
        __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkG));
        start_vec_block(2, 2, 2, 2, 2);
        dayan_wait_vec();

        configure_lsu_rows<0>(rows, kHeadDim * sizeof(float), config->q_nb[2], kHeadStride);
        __tx_lsu_dma_d2u<0>(uram_address(kChunkQ), (uint64_t) gdn_row(q, config->q_nb, head % config->q_heads, token, q_sequence));
        __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(config->k_nb[2]);
        __tx_lsu_dma_d2u<0>(uram_address(kChunkK), (uint64_t) gdn_row(k, config->k_nb, head % config->k_heads, token, k_sequence));
        __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(config->v_nb[2]);
        __tx_lsu_dma_d2u<0>(uram_address(kChunkV), (uint64_t) gdn_row(v, config->v_nb, head, token, sequence));
        configure_lsu_rows<0>(rows, 64, config->g_nb[2], sizeof(float));
        __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(sizeof(float) - 1);
        __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(sizeof(float) - 1);
        __tx_lsu_dma_d2u<0>(uram_address(kChunkG), (uint64_t) gdn_row(g, config->g_nb, head, token, sequence));
        __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(config->beta_nb[2]);
        __tx_lsu_dma_d2u<0>(uram_address(kChunkBeta), (uint64_t) gdn_row(beta, config->beta_nb, head, token, sequence));
        dayan_wait_lsu();

        configure_chunk_mat(rows, rows, kHeadDim, kHeadStride, kHeadStride, kChunkStride);
        __tx_mat_mm<0>(uram_address(kChunkKK), uram_address(kChunkK), uram_address(kChunkK));
        __tx_mat_mm<0>(uram_address(kChunkKQ), uram_address(kChunkK), uram_address(kChunkQ));
        configure_chunk_mat(kHeadDim, rows, kHeadDim, kHeadStride, kHeadStride, kChunkStride);
        __tx_mat_mm<0>(uram_address(kChunkR), uram_address(kChunkState), uram_address(kChunkK));
        __tx_mat_mm<0>(uram_address(kChunkO), uram_address(kChunkState), uram_address(kChunkQ));
        set_lsu_matrix<0, true, 2>(checksum, kChunkTokens, kHeadDim * sizeof(float), kHeadStride, kChunkStride);
        __tx_lsu_dma_u2u<0>(uram_address(kChunkTemp), uram_address(kChunkV));
        issue_chunk_prefix(rows);
        dayan_wait_mat();
        dayan_wait_lsu();
        dayan_wait_vec();
        issue_chunk_coefficients(rows, config->scale_bits);
        dayan_wait_vec();
        issue_chunk_residual(valid_mask, config->scale_bits);
        dayan_wait_vec();
        issue_chunk_solve(rows);
        dayan_wait_vec();

        set_lsu_matrix<0, true, 2>(checksum, kChunkTokens, kChunkTokens * sizeof(float), kChunkStride, kChunkStride);
        __tx_lsu_dma_u2u<0>(uram_address(kChunkTemp), uram_address(kChunkKQ));
        dayan_wait_lsu();
        configure_chunk_mat(kHeadDim, rows, rows, kChunkStride, kChunkStride, kChunkStride);
        __tx_mat_mma<0>(uram_address(kChunkO), uram_address(kChunkR), uram_address(kChunkTemp));
        dayan_wait_mat();

        set_lsu_matrix<0, true, 2>(checksum, kHeadDim, kChunkTokens * sizeof(float), kChunkStride, kHeadStride);
        __tx_lsu_dma_u2u<0>(uram_address(kChunkQ), uram_address(kChunkO));
        issue_chunk_decay(rows);
        dayan_wait_lsu();
        const uint64_t value_index = (((uint64_t) sequence * config->tokens + token) * config->heads + head) * kHeadDim;
        configure_lsu_rows<0>(rows, kHeadDim * sizeof(float), kHeadStride, (uint64_t) config->heads * kHeadDim * sizeof(float));
        __tx_lsu_dma_u2d<0>((uint64_t) (output + value_index), uram_address(kChunkQ));
        dayan_wait_vec();
        set_lsu_matrix<0, true, 2>(checksum, kChunkTokens, kHeadDim * sizeof(float), kHeadStride, kChunkStride);
        __tx_lsu_dma_u2u<0>(uram_address(kChunkTemp), uram_address(kChunkK));
        dayan_wait_lsu();
        configure_chunk_mat(kHeadDim, kHeadDim, rows, kChunkStride, kChunkStride, kHeadStride);
        __tx_mat_mma<0>(uram_address(kChunkState), uram_address(kChunkR), uram_address(kChunkTemp));
        dayan_wait_mat();
    }
    configure_lsu_rows<0>(kHeadDim, kHeadDim * sizeof(float), kHeadStride, kHeadDim * sizeof(float));
    __tx_lsu_dma_u2d<0>((uint64_t) (output + attn_count + state_index), uram_address(kChunkState));
    dayan_wait_lsu();
}

template <long LsuGroup>
static inline long run_gated_delta_net_task(const DayanGatedDeltaNetConfig * config,
                                            const float *                    q,
                                            const float *                    k,
                                            const float *                    v,
                                            const float *                    g,
                                            const float *                    beta,
                                            const float *                    state,
                                            float *                          output) {
    DAYAN_PARAMS_IF(config == nullptr || q == nullptr || k == nullptr || v == nullptr || g == nullptr ||
                    beta == nullptr || state == nullptr || output == nullptr || config->head_dim != kHeadDim ||
                    config->heads == 0 || config->tokens == 0 || config->sequences == 0 || config->snapshots == 0 ||
                    (config->gate_width != 1 && config->gate_width != kHeadDim) || config->q_heads == 0 ||
                    config->k_heads == 0 || config->heads % config->q_heads != 0 ||
                    config->heads % config->k_heads != 0 || config->q_sequences == 0 || config->k_sequences == 0 ||
                    config->sequences % config->q_sequences != 0 || config->sequences % config->k_sequences != 0 ||
                    dayan_task_mode() != 1U || dayan_task_dim(DAYAN_TASK_X) == 0U ||
                    dayan_task_dim(DAYAN_TASK_X) > kDayanGatedDeltaNetMaxTasks ||
                    dayan_task_id(DAYAN_TASK_X) >= dayan_task_dim(DAYAN_TASK_X) ||
                    dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U ||
                    dayan_uram_base() == nullptr ||
                    dayan_uram_size() < DAYAN_GATED_DELTA_NET_TASK_URAM_BYTES) {
        return -22;
    }

    const uint64_t vector_count     = (uint64_t) config->sequences * config->tokens * config->heads;
    const uint64_t state_count      = (uint64_t) config->sequences * config->heads * kHeadDim * kHeadDim;
    const uint64_t attn_count       = vector_count * kHeadDim;
    const uint64_t state_slot_count = state_count;

    *uram_pointer<float>(kUramLog2E) = 1.4426950408889634f;

    for (long group = 0; group < kRowGroups; ++group) {
        for (long lane = 0; lane < kRowsPerGroup; ++lane) {
            *uram_pointer<uint64_t>(kUramMaskTable + group * kMaskGroupBytes + lane * kScalarStride) =
                uint64_t{ 1 } << lane;
        }
    }

    configure_lsu_rows<LsuGroup>(kHeadDim, kHeadDim * sizeof(float), kHeadDim * sizeof(float),
                                 kHeadDim * sizeof(float));
    const uint32_t task       = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks      = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t work_items = (uint64_t) config->sequences * config->heads;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint32_t sequence    = (uint32_t) (work / config->heads);
        const uint32_t head        = (uint32_t) (work % config->heads);
        if (config->gate_width == 1 && config->snapshots == 1 && config->tokens >= 16) {
            run_chunked_head(config, q, k, v, g, beta, state, output, head, sequence);
            continue;
        }
        const uint64_t state_index = ((uint64_t) sequence * config->heads + head) * kHeadDim * kHeadDim;
        __tx_lsu_dma_d2u<LsuGroup>(uram_address(kUramState), reinterpret_cast<uint64_t>(state + state_index));
        dayan_wait_lsu();

        for (uint32_t token = 0; token < config->tokens; ++token) {
            const uint64_t vector_index = ((uint64_t) sequence * config->tokens + token) * config->heads + head;
            const uint64_t value_index  = vector_index * kHeadDim;
            const uint32_t q_sequence   = sequence / (config->sequences / config->q_sequences);
            const uint32_t k_sequence   = sequence / (config->sequences / config->k_sequences);
            const float *  q_row        = gdn_row(q, config->q_nb, head % config->q_heads, token, q_sequence);
            const float *  k_row        = gdn_row(k, config->k_nb, head % config->k_heads, token, k_sequence);
            const float *  v_row        = gdn_row(v, config->v_nb, head, token, sequence);
            const float *  g_row        = gdn_row(g, config->g_nb, head, token, sequence);
            const float *  beta_row     = gdn_row(beta, config->beta_nb, head, token, sequence);
            configure_lsu_rows<LsuGroup>(1, kHeadDim * sizeof(float), kHeadDim * sizeof(float),
                                         kHeadDim * sizeof(float));
            __tx_lsu_dma_d2u<LsuGroup>(uram_address(kUramQ), reinterpret_cast<uint64_t>(q_row));
            __tx_lsu_dma_d2u<LsuGroup>(uram_address(kUramK), reinterpret_cast<uint64_t>(k_row));
            __tx_lsu_dma_d2u<LsuGroup>(uram_address(kUramV), reinterpret_cast<uint64_t>(v_row));
            if (config->gate_width == kHeadDim) {
                __tx_lsu_dma_d2u<LsuGroup>(uram_address(kUramG), reinterpret_cast<uint64_t>(g_row));
            }
            configure_lsu_scalar<LsuGroup>();
            if (config->gate_width == 1) {
                __tx_lsu_dma_d2u<LsuGroup>(uram_address(kUramG), reinterpret_cast<uint64_t>(g_row));
            }
            __tx_lsu_dma_d2u<LsuGroup>(uram_address(kUramBeta), reinterpret_cast<uint64_t>(beta_row));
            dayan_wait_lsu();

            float * attn_output = output + value_index;
            if (config->gate_width == 1) {
                issue_scalar_gate(uram_address(kUramG));
                issue_scale_state<0>();
            } else {
                issue_vector_gate(uram_address(kUramG));
                issue_scale_state<kVecBytes>();
            }
            configure_vec_recurrence();
            issue_recurrence(config->scale_bits);
            dayan_wait_vec();
            configure_lsu_rows<LsuGroup>(1, kHeadDim * sizeof(float), kHeadDim * sizeof(float), kHeadDim * sizeof(float));
            __tx_lsu_dma_u2d<LsuGroup>(reinterpret_cast<uint64_t>(attn_output), uram_address(kUramOutputPacked));
            dayan_wait_lsu();

            const int64_t target_slot = (int64_t) config->tokens - 1 - token;
            if (target_slot >= 0 && target_slot < config->snapshots) {
                const uint64_t output_state_index =
                    attn_count + (uint64_t) target_slot * state_slot_count + state_index;
                configure_lsu_rows<LsuGroup>(kHeadDim, kHeadDim * sizeof(float), kHeadDim * sizeof(float),
                                             kHeadDim * sizeof(float));
                __tx_lsu_dma_u2d<LsuGroup>(reinterpret_cast<uint64_t>(output + output_state_index),
                                           uram_address(kUramState));
                dayan_wait_lsu();
            }
        }
    }

    return 0;
}

static inline long run_gated_delta_net(const DayanGatedDeltaNetConfig * config,
                                       const float *                    q,
                                       const float *                    k,
                                       const float *                    v,
                                       const float *                    g,
                                       const float *                    beta,
                                       const float *                    state,
                                       float *                          output) {
    return run_gated_delta_net_task<0>(config, q, k, v, g, beta, state, output);
}

}  // namespace ggml_dayan_gdn_detail
