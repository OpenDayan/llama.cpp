#pragma once

#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#include <stdint.h>

namespace ggml_dayan_gdn_detail {

constexpr uint32_t kHeadDim  = 128;
constexpr long     kLanes    = 64;
constexpr long     kVecBytes = kLanes * sizeof(float);

constexpr long kUramState   = 0x00000;
constexpr long kUramQ       = 0x10000;
constexpr long kUramK       = 0x10200;
constexpr long kUramV       = 0x10400;
constexpr long kUramG       = 0x10600;
constexpr long kUramGate    = 0x10800;
constexpr long kUramDot     = 0x10a00;
constexpr long kUramDelta   = 0x10c00;
constexpr long kUramBeta    = 0x10e00;
constexpr long kUramScale   = 0x10e40;
constexpr long kUramLog2E   = 0x10e80;
constexpr long kUramEnd     = 0x10ec0;

static_assert(kUramEnd <= DAYAN_GATED_DELTA_NET_TASK_URAM_BYTES, "Gated Delta Net URAM layout is too large");

enum VecTcsr {
    kVecControl = 0,
    kVecDim0    = 2,
    kVecDim1    = 3,
    kVecDim2    = 4,
    kVecDim3    = 5,
    kVecDim4    = 6,
    kVecDim5    = 7,
    kVecV0Base  = 37,
    kVecV1Base  = 44,
    kVecV2Base  = 51,
    kVecV3Base  = 58,
};

static inline uint64_t uram_address(long offset) {
    return reinterpret_cast<uint64_t>(static_cast<char *>(dayan_uram_base()) + offset);
}

static inline long mat_address(long offset) {
    return offset;
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

static inline void configure_vec_dims(long rows) {
    __tx_vec_tcsr_write<kVecDim0>(0);
    __tx_vec_tcsr_write<kVecDim1>(0);
    __tx_vec_tcsr_write<kVecDim2>(rows - 1);
    __tx_vec_tcsr_write<kVecDim3>(0);
    __tx_vec_tcsr_write<kVecDim4>(0);
    __tx_vec_tcsr_write<kVecDim5>(0);
}

static inline long vec_control(long instructions) {
    const uint64_t count = static_cast<uint64_t>(instructions) & 0xffU;
    return static_cast<long>(count | (count << 8) | (count << 16) | (count << 24) | (count << 32));
}

static inline void begin_vec_block(long instructions) {
    long free_entries;
    do {
        __tx_vec_tcsr_read<kVecControl>(free_entries);
    } while (free_entries < instructions);
}

static inline void start_vec_block(long instructions) {
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
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

template <long Group> static inline void configure_mat_recurrence() {
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_M>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_N>(kHeadDim - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_K>(kHeadDim - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASTRIDE>(kHeadDim * sizeof(float));
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSTRIDE>(kHeadDim * sizeof(float));
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSTRIDE>(kHeadDim * sizeof(float));
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DATATYPE>(BaseNPUInstBuilder::DT_FP32);
}

template <long Group> static inline void issue_mat_recurrence(long input) {
    __tx_mat_mm<Group>(mat_address(kUramDot), mat_address(input), mat_address(kUramState));
}

static inline void issue_gate(uint64_t gate_input, bool scalar_gate) {
    constexpr long instructions = 5;

    configure_vec_dims(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    if (scalar_gate) {
        __tx_vec_lsu_u2v_broadcast32<0, 0>(gate_input);
    } else {
        __tx_vec_lsu_u2v<0, 0>(gate_input);
    }
    __tx_vec_lsu_u2v_broadcast32<1, 1>(uram_address(kUramLog2E));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    __tx_vec_lsu_v2u<3, 0>(uram_address(kUramGate));
    start_vec_block(instructions);
}

static inline void issue_scale_state(long half) {
    constexpr long instructions = 4;
    const long     half_offset  = half * kVecBytes;

    configure_vec_dims(kHeadDim);
    configure_vec_iterator<kVecV0Base>(0, 0, kHeadDim * sizeof(float), 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kHeadDim * sizeof(float), 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_address(kUramState + half_offset));
    __tx_vec_lsu_u2v<1, 1>(uram_address(kUramGate));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kUramState + half_offset));
    start_vec_block(instructions);
}

static inline void issue_update_state(long half) {
    constexpr long instructions = 6;
    const long     half_offset  = half * kVecBytes;

    configure_vec_dims(kHeadDim);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, sizeof(float), 0);
    configure_vec_iterator<kVecV2Base>(0, 0, kHeadDim * sizeof(float), 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_address(kUramK + half_offset));
    __tx_vec_lsu_u2v_broadcast32<1, 1>(uram_address(kUramDelta));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 2>(uram_address(kUramState + half_offset));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<2, 0>(uram_address(kUramState + half_offset));
    start_vec_block(instructions);
}

static inline void issue_output() {
    constexpr long instructions = 4;

    configure_vec_dims(2);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_address(kUramDot));
    __tx_vec_lsu_u2v_broadcast32<1, 1>(uram_address(kUramScale));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(uram_address(kUramDot));
    start_vec_block(instructions);
}

static inline void issue_delta() {
    constexpr long instructions = 6;

    configure_vec_dims(2);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_address(kUramV));
    __tx_vec_lsu_u2v<1, 0>(uram_address(kUramDot));
    __tx_vec_vv_fp32_sub<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(uram_address(kUramBeta));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(uram_address(kUramDelta));
    start_vec_block(instructions);
}

template <long Group> static inline void configure_lsu_scalar() {
    configure_lsu_rows<Group>(1, 64, 64, 64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(sizeof(float) - 1);
}

static inline float bits_to_float(uint32_t bits) {
    union {
        uint32_t u;
        float    f;
    } value;

    value.u = bits;
    return value.f;
}

static inline const float * gdn_row(const float *  base,
                                    const uint64_t nb[4],
                                    uint32_t       head,
                                    uint32_t       token,
                                    uint32_t       sequence) {
    return reinterpret_cast<const float *>(reinterpret_cast<const char *>(base) + (uint64_t) head * nb[1] +
                                           (uint64_t) token * nb[2] + (uint64_t) sequence * nb[3]);
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
                    dayan_task_dim(DAYAN_TASK_X) > kDayanNativeMaxTasks ||
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

    *uram_pointer<float>(kUramScale) = bits_to_float(config->scale_bits);
    dayan_cache_flush(uram_pointer<float>(kUramScale), 64);
    dayan_cache_flush(uram_pointer<float>(kUramLog2E), 64);

    configure_lsu_rows<LsuGroup>(kHeadDim, kHeadDim * sizeof(float), kHeadDim * sizeof(float),
                                 kHeadDim * sizeof(float));
    configure_mat_recurrence<LsuGroup>();
    const uint32_t task       = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks      = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t work_items = (uint64_t) config->sequences * config->heads;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint32_t sequence    = (uint32_t) (work / config->heads);
        const uint32_t head        = (uint32_t) (work % config->heads);
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

            if (config->gate_width == 1) {
                issue_gate(uram_address(kUramG), true);
                dayan_wait_vec();
            }
            for (long half = 0; half < 2; ++half) {
                if (config->gate_width == kHeadDim) {
                    issue_gate(uram_address(kUramG + half * kVecBytes), false);
                    dayan_wait_vec();
                }
                issue_scale_state(half);
                dayan_wait_vec();
            }

            issue_mat_recurrence<LsuGroup>(kUramK);
            dayan_wait_mat();
            issue_delta();
            dayan_wait_vec();

            for (long half = 0; half < 2; ++half) {
                issue_update_state(half);
                dayan_wait_vec();
            }

            float * attn_output = output + value_index;
            issue_mat_recurrence<LsuGroup>(kUramQ);
            dayan_wait_mat();
            issue_output();
            dayan_wait_vec();
            configure_lsu_rows<LsuGroup>(1, kHeadDim * sizeof(float), kHeadDim * sizeof(float),
                                         kHeadDim * sizeof(float));
            __tx_lsu_dma_u2d<LsuGroup>(reinterpret_cast<uint64_t>(attn_output), uram_address(kUramDot));
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
