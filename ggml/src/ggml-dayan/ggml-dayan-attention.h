#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

#    include <dayan_sys.h>

namespace ggml_dayan_matmul_detail {

enum : uint64_t {
    kAttentionLsuMaxDimension = 1ULL << 24,
    kAttentionLsuMaxStride    = (1ULL << 28) - 1U,
    kAttentionMaxHeadDim = Constants::kTileK,
    kAttentionFloatBytes = kAttentionMaxHeadDim * sizeof(float),
    kAttentionLowpBytes  = kAttentionMaxHeadDim * sizeof(uint16_t),
    kAttentionPairBytes  = kAttentionFloatBytes / 2,
    kAttentionRaw        = 0,
    kAttentionNorm       = kAttentionRaw + kAttentionFloatBytes,
    kAttentionNormalized = kAttentionNorm + kAttentionFloatBytes,
    kAttentionRotated    = kAttentionNormalized + kAttentionFloatBytes,
    kAttentionLowp       = kAttentionRotated + kAttentionFloatBytes,
    kAttentionX0         = kAttentionLowp + kAttentionLowpBytes,
    kAttentionX1         = kAttentionX0 + kAttentionPairBytes,
    kAttentionFactors    = kAttentionX1 + kAttentionPairBytes,
    kAttentionPosition   = kAttentionFactors + kAttentionPairBytes,
    kAttentionTheta      = kAttentionPosition + kAttentionPairBytes,
    kAttentionSine       = kAttentionTheta + kAttentionPairBytes,
    kAttentionCosine     = kAttentionSine + kAttentionPairBytes,
    kAttentionOutput0    = kAttentionCosine + kAttentionPairBytes,
    kAttentionOutput1    = kAttentionOutput0 + kAttentionPairBytes,
    kAttentionLane       = kAttentionOutput1 + kAttentionPairBytes,
    kAttentionAxes       = kAttentionLane + Constants::kVecBytes,
    kAttentionPartial    = kAttentionAxes + kAttentionPairBytes,
    kAttentionPositions  = kAttentionPartial + kAttentionFloatBytes,
    kAttentionConstants  = kAttentionPositions + Constants::kVecBytes,
    kAttentionEnd        = kAttentionConstants + Constants::kVecBytes,
};

enum AttentionConstant : uint32_t {
    kAttentionEpsilon,
    kAttentionInvHeadDim,
    kAttentionThetaScale,
    kAttentionFreqScale,
    kAttentionFactor,
    kAttentionPairBase,
    kAttentionAxis0,
    kAttentionAxis1,
    kAttentionAxis2,
};

static_assert(kAttentionEnd <= MATMUL_DAYAN_1C_URAM_BYTES, "Attention scratch exceeds kernel URAM");

static inline uint64_t attention_uram(uint64_t offset) {
    return reinterpret_cast<uint64_t>(static_cast<char *>(dayan_uram_base()) + offset);
}

template <typename T> static inline T * attention_uram_pointer(uint64_t offset) {
    return reinterpret_cast<T *>(attention_uram(offset));
}

template <long Group>
static inline void configure_attention_lsu_3d(uint32_t dim0,
                                              uint64_t input_dim0_stride,
                                              uint64_t output_dim0_stride,
                                              uint32_t dim1,
                                              uint64_t input_dim1_stride,
                                              uint64_t output_dim1_stride,
                                              uint32_t dim2,
                                              uint64_t input_dim2_stride,
                                              uint64_t output_dim2_stride,
                                              uint32_t read_bytes,
                                              uint32_t write_bytes) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(dim0 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(dim1 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(dim2 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(input_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(input_dim2_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(output_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(output_dim2_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(read_bytes - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(write_bytes - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group>
static inline void configure_attention_lsu(uint32_t transactions,
                                           uint64_t input_stride,
                                           uint64_t output_stride,
                                           uint32_t read_bytes,
                                           uint32_t write_bytes) {
    configure_attention_lsu_3d<Group>(transactions, input_stride, output_stride, 1, 0, 0, 1, 0, 0, read_bytes,
                                      write_bytes);
}

static inline void attention_load_bytes(uint64_t destination, const void * source, uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    bool           issued = false;

    if (full != 0U) {
        configure_attention_lsu<0>(full, 64, 64, 64, 64);
        __tx_lsu_dma_d2u<0>(destination, reinterpret_cast<uint64_t>(source));
        issued = true;
    }
    if (tail != 0U) {
        configure_attention_lsu<0>(1, 0, 0, tail, tail);
        __tx_lsu_dma_d2u<0>(destination + (uint64_t) full * 64U,
                            reinterpret_cast<uint64_t>(source) + (uint64_t) full * 64U);
        issued = true;
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline void attention_store_bytes(void * destination, uint64_t source, uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    bool           issued = false;

    if (full != 0U) {
        configure_attention_lsu<0>(full, 64, 64, 64, 64);
        __tx_lsu_dma_u2d<0>(reinterpret_cast<uint64_t>(destination), source);
        issued = true;
    }
    if (tail != 0U) {
        configure_attention_lsu<0>(1, 0, 0, tail, tail);
        __tx_lsu_dma_u2d<0>(reinterpret_cast<uint64_t>(destination) + (uint64_t) full * 64U,
                            source + (uint64_t) full * 64U);
        issued = true;
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline void attention_copy_bytes(void * destination, const void * source, uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    bool           issued = false;

    if (full != 0U) {
        configure_attention_lsu<0>(full, 64, 64, 64, 64);
        __tx_lsu_dma_d2d<0>(reinterpret_cast<uint64_t>(destination), reinterpret_cast<uint64_t>(source));
        issued = true;
    }
    if (tail != 0U) {
        configure_attention_lsu<0>(1, 0, 0, tail, tail);
        __tx_lsu_dma_d2d<0>(reinterpret_cast<uint64_t>(destination) + (uint64_t) full * 64U,
                            reinterpret_cast<uint64_t>(source) + (uint64_t) full * 64U);
        issued = true;
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline void attention_copy_uram(uint64_t destination,
                                       uint64_t source,
                                       uint32_t elements,
                                       uint64_t input_stride,
                                       uint64_t output_stride) {
    configure_attention_lsu<0>(elements, input_stride, output_stride, sizeof(float), sizeof(float));
    __tx_lsu_dma_u2u<0>(destination, source);
    dayan_wait_lsu();
}

static inline void attention_copy_uram_pair(uint64_t destination,
                                            uint64_t source,
                                            uint32_t elements,
                                            uint64_t input_stride,
                                            uint64_t output_stride,
                                            uint64_t input_pair_stride,
                                            uint64_t output_pair_stride) {
    configure_attention_lsu_3d<0>(elements, input_stride, output_stride, 2, input_pair_stride, output_pair_stride, 1, 0,
                                  0, sizeof(float), sizeof(float));
    __tx_lsu_dma_u2u<0>(destination, source);
    dayan_wait_lsu();
}

static inline void attention_load_i32(uint64_t destination,
                                      const int32_t * source,
                                      uint32_t count,
                                      uint64_t input_stride) {
    configure_attention_lsu<0>(count, input_stride, sizeof(int32_t), sizeof(int32_t), sizeof(int32_t));
    __tx_lsu_dma_d2u<0>(destination, reinterpret_cast<uint64_t>(source));
    dayan_wait_lsu();
}

static inline void attention_reset_scratch() {
    uint32_t * zero = attention_uram_pointer<uint32_t>(kAttentionRaw);
    for (uint32_t index = 0; index < 16; ++index) {
        zero[index] = 0;
    }
    dayan_cache_flush(zero, 64);
    configure_attention_lsu<0>(kAttentionEnd / 64U, 0, 64, 64, 64);
    __tx_lsu_dma_u2u<0>(attention_uram(kAttentionRaw), attention_uram(kAttentionRaw));
    dayan_wait_lsu();
}

static inline void attention_initialize_constants(const DayanAttentionProjectionConfig * config) {
    uint32_t * constants = attention_uram_pointer<uint32_t>(kAttentionConstants);
    float *    values    = reinterpret_cast<float *>(constants);

    constants[kAttentionEpsilon]    = config->norm_eps_bits;
    constants[kAttentionThetaScale] = config->theta_scale_bits;
    constants[kAttentionFreqScale]  = config->freq_scale_bits;
    constants[kAttentionFactor]     = config->attn_factor_bits;
    values[kAttentionInvHeadDim]    = 1.0f / (float) config->head_dim;
    constants[kAttentionAxis0]      = 0;
    constants[kAttentionAxis1]      = 1;
    constants[kAttentionAxis2]      = 2;

    int32_t * lane = attention_uram_pointer<int32_t>(kAttentionLane);
    for (uint32_t index = 0; index < Constants::kVecLanes; ++index) {
        lane[index] = (int32_t) index;
    }

    const int32_t section_count = config->sections[0] + config->sections[1] + config->sections[2] + config->sections[3];
    int32_t *     axes          = attention_uram_pointer<int32_t>(kAttentionAxes);
    for (uint32_t pair = 0; pair < config->n_rot / 2U; ++pair) {
        int32_t axis = 0;
        if (config->rope_mode == 8U && section_count > 0) {
            const int32_t sector = (int32_t) pair % section_count;
            axis = sector >= config->sections[0];
            axis += sector >= config->sections[0] + config->sections[1];
            axis += sector >= config->sections[0] + config->sections[1] + config->sections[2];
        }
        axes[pair] = axis;
    }
    dayan_cache_flush(lane, Constants::kVecBytes);
    dayan_cache_flush(axes, kAttentionPairBytes);
    dayan_cache_flush(constants, Constants::kVecBytes);
}

template <typename Format> static inline void issue_attention_lowp_to_f32(uint64_t input, uint32_t chunks) {
    constexpr long instructions = 3;

    ggml_dayan_vec_detail::configure_vec_grid(chunks);
    configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    if constexpr (Format::kMatDataType == BaseNPUInstBuilder::DT_BF16) {
        ggml_dayan_vec_detail::vec_cvt_fp32_bf16<0, 0, 0>();
    } else {
        ggml_dayan_vec_detail::vec_cvt_fp32_fp16<0, 0, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>(input);
    start_vec_block(instructions);
}

static inline void issue_attention_square_sum(uint64_t input, uint64_t partial, uint32_t chunks) {
    constexpr long instructions = 15;

    ggml_dayan_vec_detail::configure_vec_grid(chunks);
    configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    __tx_vec_shuffle_xor<1, 0>(32);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_shuffle_xor<1, 0>(16);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_shuffle_xor<1, 0>(8);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_shuffle_xor<1, 0>(4);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_shuffle_xor<1, 0>(2);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_shuffle_xor<1, 0>(1);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(partial);
    start_vec_block(instructions);
}

static inline void issue_attention_inverse_rms(uint64_t partial, uint64_t inverse, uint32_t chunks) {
    const long instructions = 7 + 2 * (chunks - 1U);

    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(partial);
    for (uint32_t chunk = 1; chunk < chunks; ++chunk) {
        __tx_vec_lsu_u2v<1, 1>(partial + (uint64_t) chunk * Constants::kVecBytes);
        __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    }
    __tx_vec_lsu_u2v_broadcast32<1, 1>(attention_uram(kAttentionConstants) +
                                        kAttentionInvHeadDim * sizeof(uint32_t));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(attention_uram(kAttentionConstants) +
                                        kAttentionEpsilon * sizeof(uint32_t));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 4>();
    __tx_vec_lsu_v2u<0, 3>(inverse);
    start_vec_block(instructions);
}

static inline void issue_attention_normalize(uint64_t input,
                                             uint64_t norm,
                                             uint64_t inverse,
                                             uint64_t output,
                                             uint32_t chunks) {
    constexpr long instructions = 6;

    ggml_dayan_vec_detail::configure_vec_grid(chunks);
    configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, Constants::kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    __tx_vec_lsu_u2v<1, 1>(norm);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(inverse);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output);
    start_vec_block(instructions);
}

static inline void issue_attention_factors(uint64_t output, uint32_t pair_base) {
    constexpr long instructions = 11;
    float *        constants    = attention_uram_pointer<float>(kAttentionConstants);
    constants[kAttentionPairBase] = (float) pair_base;
    dayan_cache_flush(constants, Constants::kVecBytes);

    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(attention_uram(kAttentionLane));
    ggml_dayan_vec_detail::vec_cvt_fp32_int32<0, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(attention_uram(kAttentionConstants) +
                                        kAttentionPairBase * sizeof(uint32_t));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(attention_uram(kAttentionConstants) +
                                        kAttentionThetaScale * sizeof(uint32_t));
    __tx_vec_v_fp32_sfu<1, 16, 0>();
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(attention_uram(kAttentionConstants) +
                                        kAttentionFreqScale * sizeof(uint32_t));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output);
    start_vec_block(instructions);
}

static inline void emit_attention_axis_select(uint64_t axes, uint64_t axis_value, uint64_t position) {
    __tx_vec_lsu_u2v<1, 0>(axes);
    __tx_vec_lsu_u2v_broadcast32<1, 1>(axis_value);
    __tx_vec_vv_int32_cmp<16, 17, 0, 0b010>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(position);
    __tx_vec_vv_mux2<0, 0, 16, 0>();
}

static inline void issue_attention_position(uint64_t axes,
                                            uint64_t positions,
                                            uint64_t output,
                                            bool mrope) {
    const long instructions = mrope ? 18 : 3;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    if (mrope) {
        __tx_vec_lsu_u2v_broadcast32<0, 0>(positions + 3U * sizeof(int32_t));
        emit_attention_axis_select(axes, attention_uram(kAttentionConstants) +
                                             kAttentionAxis2 * sizeof(uint32_t),
                                   positions + 2U * sizeof(int32_t));
        emit_attention_axis_select(axes, attention_uram(kAttentionConstants) +
                                             kAttentionAxis1 * sizeof(uint32_t),
                                   positions + sizeof(int32_t));
        emit_attention_axis_select(axes, attention_uram(kAttentionConstants) +
                                             kAttentionAxis0 * sizeof(uint32_t),
                                   positions);
    } else {
        __tx_vec_lsu_u2v_broadcast32<0, 0>(positions);
    }
    ggml_dayan_vec_detail::vec_cvt_fp32_int32<0, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output);
    start_vec_block(instructions);
}

static inline void issue_attention_theta(uint64_t position, uint64_t factors, uint64_t output) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(position);
    __tx_vec_lsu_u2v<1, 1>(factors);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output);
    start_vec_block(instructions);
}

static inline void issue_attention_sincos(uint64_t theta, uint64_t sine, uint64_t cosine) {
    constexpr long instructions = 10;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(theta);
    __tx_vec_v_fp32_sfu<0, 0, 2>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(attention_uram(kAttentionConstants) +
                                        kAttentionFactor * sizeof(uint32_t));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(sine);
    __tx_vec_lsu_u2v<0, 0>(theta);
    __tx_vec_v_fp32_sfu<0, 0, 3>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(attention_uram(kAttentionConstants) +
                                        kAttentionFactor * sizeof(uint32_t));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(cosine);
    start_vec_block(instructions);
}

static inline void issue_attention_rotation(uint64_t x0,
                                            uint64_t x1,
                                            uint64_t sine,
                                            uint64_t cosine,
                                            uint64_t output0,
                                            uint64_t output1) {
    constexpr long instructions = 16;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(x0);
    __tx_vec_lsu_u2v<1, 1>(cosine);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(x1);
    __tx_vec_lsu_u2v<1, 1>(sine);
    __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>(output0);
    __tx_vec_lsu_u2v<0, 0>(x0);
    __tx_vec_lsu_u2v<1, 1>(sine);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(x1);
    __tx_vec_lsu_u2v<1, 1>(cosine);
    __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output1);
    start_vec_block(instructions);
}

static inline bool attention_projection_config_valid(const DayanAttentionProjectionConfig * config, bool has_gate) {
    if (config == nullptr || config->tokens == 0 || config->heads == 0 || config->head_dim == 0 || config->n_rot == 0 ||
        config->head_dim > kAttentionMaxHeadDim || config->n_rot > config->head_dim || (config->n_rot & 1U) != 0 ||
        (config->rope_mode != 0U && config->rope_mode != 2U && config->rope_mode != 8U)) {
        return false;
    }
    const uint64_t projection_width = (uint64_t) config->heads * config->head_dim * (has_gate ? 2U : 1U);
    return config->matmul.m == config->tokens && config->matmul.n == projection_width &&
           config->matmul.output_stride >= projection_width * sizeof(uint16_t);
}

template <typename Format>
static inline void run_projection_matmul(const DayanMatmulConfig * config,
                                         const uint16_t *          activation,
                                         const uint16_t *          weight,
                                         uint16_t *                output,
                                         DayanMatmulPerf *         perf) {
    run_matmul_kernel<Format, 1, sizeof(uint16_t), true, epilogue_identity_bf16<0>, epilogue_identity_bf16<4>,
                      epilogue_identity_bf16<2>, epilogue_identity_bf16<0>, epilogue_identity_bf16<4>,
                      epilogue_identity_bf16<2>>(config, activation, weight, nullptr, output, perf);
}

template <>
inline void run_projection_matmul<Fp16Format>(const DayanMatmulConfig * config,
                                              const uint16_t *          activation,
                                              const uint16_t *          weight,
                                              uint16_t *                output,
                                              DayanMatmulPerf *         perf) {
    run_matmul_kernel<Fp16Format, 1, sizeof(uint16_t), true, epilogue_identity_fp16<0>, epilogue_identity_fp16<4>,
                      epilogue_identity_fp16<2>, epilogue_identity_fp16<0>, epilogue_identity_fp16<4>,
                      epilogue_identity_fp16<2>>(config, activation, weight, nullptr, output, perf);
}

template <typename Format>
static inline void postprocess_attention_head(const DayanAttentionProjectionConfig * config,
                                              const uint16_t *                       source,
                                              uint16_t *                             destination) {
    const uint32_t chunks = (config->head_dim + Constants::kVecLanes - 1U) / Constants::kVecLanes;
    configure_attention_lsu<0>(config->head_dim, sizeof(uint16_t), sizeof(float), sizeof(uint16_t), sizeof(float));
    __tx_lsu_dma_d2u<0>(attention_uram(kAttentionRaw), reinterpret_cast<uint64_t>(source));
    dayan_wait_lsu();
    issue_attention_lowp_to_f32<Format>(attention_uram(kAttentionRaw), chunks);
    dayan_wait_vec();
    issue_attention_square_sum(attention_uram(kAttentionRaw), attention_uram(kAttentionPartial), chunks);
    dayan_wait_vec();
    issue_attention_inverse_rms(attention_uram(kAttentionPartial), attention_uram(kAttentionPartial), chunks);
    dayan_wait_vec();
    issue_attention_normalize(attention_uram(kAttentionRaw), attention_uram(kAttentionNorm),
                              attention_uram(kAttentionPartial), attention_uram(kAttentionNormalized), chunks);
    dayan_wait_vec();

    const uint32_t half = config->n_rot / 2U;
    for (uint32_t pair_base = 0; pair_base < half; pair_base += Constants::kVecLanes) {
        const uint32_t pair_count =
            half - pair_base < Constants::kVecLanes ? half - pair_base : Constants::kVecLanes;
        issue_attention_factors(attention_uram(kAttentionFactors), pair_base);
        dayan_wait_vec();
        issue_attention_position(attention_uram(kAttentionAxes) + (uint64_t) pair_base * sizeof(int32_t),
                                 attention_uram(kAttentionPositions), attention_uram(kAttentionPosition),
                                 config->rope_mode == 8U);
        dayan_wait_vec();
        issue_attention_theta(attention_uram(kAttentionPosition), attention_uram(kAttentionFactors),
                              attention_uram(kAttentionTheta));
        dayan_wait_vec();
        issue_attention_sincos(attention_uram(kAttentionTheta), attention_uram(kAttentionSine),
                               attention_uram(kAttentionCosine));
        dayan_wait_vec();

        if (config->rope_mode == 0U) {
            attention_copy_uram_pair(attention_uram(kAttentionX0),
                                     attention_uram(kAttentionNormalized) +
                                         (uint64_t) 2U * pair_base * sizeof(float),
                                     pair_count, 2U * sizeof(float), sizeof(float), sizeof(float),
                                     kAttentionPairBytes);
        } else {
            attention_copy_uram_pair(attention_uram(kAttentionX0),
                                     attention_uram(kAttentionNormalized) + (uint64_t) pair_base * sizeof(float),
                                     pair_count, sizeof(float), sizeof(float), (uint64_t) half * sizeof(float),
                                     kAttentionPairBytes);
        }
        issue_attention_rotation(attention_uram(kAttentionX0), attention_uram(kAttentionX1),
                                 attention_uram(kAttentionSine), attention_uram(kAttentionCosine),
                                 attention_uram(kAttentionOutput0), attention_uram(kAttentionOutput1));
        dayan_wait_vec();
        if (config->rope_mode == 0U) {
            attention_copy_uram_pair(attention_uram(kAttentionRotated) +
                                         (uint64_t) 2U * pair_base * sizeof(float),
                                     attention_uram(kAttentionOutput0), pair_count, sizeof(float), 2U * sizeof(float),
                                     kAttentionPairBytes, sizeof(float));
        } else {
            attention_copy_uram_pair(attention_uram(kAttentionRotated) + (uint64_t) pair_base * sizeof(float),
                                     attention_uram(kAttentionOutput0), pair_count, sizeof(float), sizeof(float),
                                     kAttentionPairBytes, (uint64_t) half * sizeof(float));
        }
    }

    if (config->n_rot < config->head_dim) {
        attention_copy_uram(attention_uram(kAttentionRotated) + (uint64_t) config->n_rot * sizeof(float),
                            attention_uram(kAttentionNormalized) + (uint64_t) config->n_rot * sizeof(float),
                            config->head_dim - config->n_rot, sizeof(float), sizeof(float));
    }
    const ggml_dayan_vec_detail::F32Tensor rotated = {
        attention_uram(kAttentionRotated), (long) config->head_dim * (long) sizeof(float)
    };
    ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
        rotated, rotated,
        { attention_uram(kAttentionLowp), (long) config->head_dim * (long) sizeof(uint16_t) }, 1,
        config->head_dim);
    attention_store_bytes(destination, attention_uram(kAttentionLowp), config->head_dim * sizeof(uint16_t));
}

template <typename Format, bool HasGate>
static inline void postprocess_attention_projection(const DayanAttentionProjectionConfig * config,
                                                    const float *                          norm,
                                                    const int32_t *                        positions,
                                                    const uint16_t *                       raw_output,
                                                    uint16_t *                             output) {
    const uint32_t head_dim        = config->head_dim;
    const uint32_t projection_step = head_dim * (HasGate ? 2U : 1U);

    attention_reset_scratch();
    attention_initialize_constants(config);
    attention_load_bytes(attention_uram(kAttentionNorm), norm, head_dim * sizeof(float));

    for (uint32_t token = 0; token < config->tokens; ++token) {
        const uint32_t position_axes = config->rope_mode == 8U ? 4U : 1U;
        attention_load_i32(attention_uram(kAttentionPositions), positions + token, position_axes,
                           (uint64_t) config->tokens * sizeof(int32_t));
        for (uint32_t head = 0; head < config->heads; ++head) {
            const uint16_t * source = raw_output + (uint64_t) token * config->matmul.output_stride / sizeof(uint16_t) +
                                      head * projection_step;
            uint16_t *       destination = output + ((uint64_t) head * config->tokens + token) * head_dim;
            postprocess_attention_head<Format>(config, source, destination);
        }
    }
}

template <typename Format, bool HasGate>
static inline void run_attention_projection(const DayanAttentionProjectionConfig * config,
                                            const uint16_t *                       activation,
                                            const uint16_t *                       weight,
                                            const float *                          norm,
                                            const int32_t *                        positions,
                                            uint16_t *                             raw_output,
                                            uint16_t *                             output,
                                            DayanMatmulPerf *                      perf) {
    if (!attention_projection_config_valid(config, HasGate) || activation == nullptr || weight == nullptr ||
        norm == nullptr || positions == nullptr || raw_output == nullptr || output == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    run_projection_matmul<Format>(&config->matmul, activation, weight, raw_output, perf);
    postprocess_attention_projection<Format, HasGate>(config, norm, positions, raw_output, output);
}

template <typename Format>
static inline void run_value_projection(const DayanAttentionProjectionConfig * config,
                                        const uint16_t *                       activation,
                                        const uint16_t *                       weight,
                                        uint16_t *                             raw_output,
                                        uint16_t *                             output,
                                        DayanMatmulPerf *                      perf) {
    if (!attention_projection_config_valid(config, false) || activation == nullptr || weight == nullptr ||
        raw_output == nullptr || output == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    const uint32_t row_bytes = config->head_dim * sizeof(uint16_t);
    const uint64_t output_head_stride = (uint64_t) config->tokens * row_bytes;
    if (config->tokens > kAttentionLsuMaxDimension || config->heads > kAttentionLsuMaxDimension ||
        config->matmul.output_stride > kAttentionLsuMaxStride || output_head_stride > kAttentionLsuMaxStride) {
        DAYAN_PARAMS_ERROR();
    }

    run_projection_matmul<Format>(&config->matmul, activation, weight, raw_output, perf);
    const uint32_t full_blocks = row_bytes / 64U;
    const uint32_t tail        = row_bytes % 64U;
    bool           issued = false;
    if (full_blocks != 0U) {
        configure_attention_lsu_3d<0>(full_blocks, 64, 64, config->heads, row_bytes, output_head_stride,
                                      config->tokens, config->matmul.output_stride, row_bytes, 64, 64);
        __tx_lsu_dma_d2d<0>(reinterpret_cast<uint64_t>(output), reinterpret_cast<uint64_t>(raw_output));
        issued = true;
    }
    if (tail != 0U) {
        const uint64_t offset = (uint64_t) full_blocks * 64U;
        configure_attention_lsu_3d<0>(1, 0, 0, config->heads, row_bytes, output_head_stride, config->tokens,
                                      config->matmul.output_stride, row_bytes, tail, tail);
        __tx_lsu_dma_d2d<0>(reinterpret_cast<uint64_t>(output) + offset,
                            reinterpret_cast<uint64_t>(raw_output) + offset);
        issued = true;
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline bool kv_projection_config_valid(const DayanKVProjectionConfig * config) {
    return config != nullptr && config->cache_rows > 0 && attention_projection_config_valid(&config->projection, false);
}

template <typename Format>
static inline void run_k_cache_projection(const DayanKVProjectionConfig * config,
                                          const uint16_t *                activation,
                                          const uint16_t *                weight,
                                          const float *                   norm,
                                          const int32_t *                 positions_indices,
                                          uint16_t *                      workspace,
                                          uint16_t *                      cache,
                                          DayanMatmulPerf *               perf) {
    if (!kv_projection_config_valid(config) || activation == nullptr || weight == nullptr || norm == nullptr ||
        positions_indices == nullptr || workspace == nullptr || cache == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    const DayanAttentionProjectionConfig * projection    = &config->projection;
    const uint32_t                         position_axes = projection->rope_mode == 8U ? 4U : 1U;
    const int32_t *                        indices = positions_indices + (uint64_t) position_axes * projection->tokens;
    run_projection_matmul<Format>(&projection->matmul, activation, weight, workspace, perf);
    attention_reset_scratch();
    attention_initialize_constants(projection);
    attention_load_bytes(attention_uram(kAttentionNorm), norm, projection->head_dim * sizeof(float));
    const uint32_t cache_width = projection->heads * projection->head_dim;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        attention_load_i32(attention_uram(kAttentionPositions), indices + token, 1, sizeof(int32_t));
        dayan_cache_invalidate(attention_uram_pointer<int32_t>(kAttentionPositions), 64);
        const int32_t cache_row = *attention_uram_pointer<int32_t>(kAttentionPositions);
        if (cache_row < 0 || (uint32_t) cache_row >= config->cache_rows) {
            DAYAN_PARAMS_ERROR();
        }
        attention_load_i32(attention_uram(kAttentionPositions), positions_indices + token, position_axes,
                           (uint64_t) projection->tokens * sizeof(int32_t));
        for (uint32_t head = 0; head < projection->heads; ++head) {
            const uint16_t * source = workspace +
                                      (uint64_t) token * projection->matmul.output_stride / sizeof(uint16_t) +
                                      head * projection->head_dim;
            uint16_t * destination  = cache + (uint64_t) cache_row * cache_width + head * projection->head_dim;
            postprocess_attention_head<Format>(projection, source, destination);
        }
    }
}

template <typename Format>
static inline void run_v_cache_projection(const DayanKVProjectionConfig * config,
                                          const uint16_t *                activation,
                                          const uint16_t *                weight,
                                          const int32_t *                 indices,
                                          uint16_t *                      workspace,
                                          uint16_t *                      cache,
                                          DayanMatmulPerf *               perf) {
    if (!kv_projection_config_valid(config) || activation == nullptr || weight == nullptr || indices == nullptr ||
        workspace == nullptr || cache == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    const DayanAttentionProjectionConfig * projection = &config->projection;
    run_projection_matmul<Format>(&projection->matmul, activation, weight, workspace, perf);
    const uint32_t cache_width = projection->heads * projection->head_dim;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        attention_load_i32(attention_uram(kAttentionPositions), indices + token, 1, sizeof(int32_t));
        dayan_cache_invalidate(attention_uram_pointer<int32_t>(kAttentionPositions), 64);
        const int32_t cache_row = *attention_uram_pointer<int32_t>(kAttentionPositions);
        if (cache_row < 0 || (uint32_t) cache_row >= config->cache_rows) {
            DAYAN_PARAMS_ERROR();
        }
        const uint16_t * source = workspace +
                                  (uint64_t) token * projection->matmul.output_stride / sizeof(uint16_t);
        uint16_t * destination = cache + (uint64_t) cache_row * cache_width;
        attention_copy_bytes(destination, source, cache_width * sizeof(uint16_t));
    }
}

template <long Group>
static inline void configure_gated_lsu(long block,
                                       long dim0,
                                       long dim1,
                                       long input_block_stride,
                                       long input_dim0_stride,
                                       long input_dim1_stride,
                                       long output_block_stride,
                                       long output_dim0_stride,
                                       long output_dim1_stride,
                                       long transpose,
                                       long read_size,
                                       long write_size) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(block);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(dim0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(dim1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(input_block_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(input_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(output_block_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(output_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(transpose);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(block);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(read_size);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(write_size);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <typename Format> static inline void issue_gated_lowp_to_f32(uint64_t input, long rows, long chunks) {
    constexpr long instructions = 3;
    constexpr long row_stride   = Constants::kTileK * sizeof(float);

    ggml_dayan_vec_detail::configure_vec_grid(chunks, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, row_stride);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, row_stride);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    if constexpr (Format::kMatDataType == BaseNPUInstBuilder::DT_BF16) {
        ggml_dayan_vec_detail::vec_cvt_fp32_bf16<0, 0, 0>();
    } else {
        ggml_dayan_vec_detail::vec_cvt_fp32_fp16<0, 0, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>(input);
    start_vec_block(instructions);
}

template <typename Format>
static inline void convert_gated_f32_to_lowp(uint64_t input, uint64_t output, long rows, long cols) {
    const ggml_dayan_vec_detail::F32Tensor tile = {
        input, Constants::kTileK * (int) sizeof(float)
    };
    ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, Constants::kLsuShared>(
        tile, tile,
        { output, Constants::kTileK * (int) sizeof(uint16_t) }, rows, cols);
}

struct GatedInputContext {
    const DayanGatedProjectionConfig * config;
    const float *                      attention;
    const uint16_t *                   gate;
};

enum : uint64_t {
    kGatedInputMatrixBytes  = Constants::kTileM * Constants::kTileK * sizeof(float),
    kGatedInputScratchBytes = 2 * kGatedInputMatrixBytes + Constants::kConstantStride,
};

template <typename Format> static int load_gated_input(const MatmulInputArgs * args) {
    const GatedInputContext *          context        = (const GatedInputContext *) args->context;
    const DayanGatedProjectionConfig * config         = context->config;
    const uint64_t                     attention_uram = args->scratch;
    const uint64_t                     gate_uram      = attention_uram + kGatedInputMatrixBytes;
    const uint64_t                     constants      = gate_uram + kGatedInputMatrixBytes;
    const uint32_t                     width          = config->heads * config->head_dim;
    const uint32_t                     heads          = args->cols / config->head_dim;
    const uint32_t                     first_head     = args->k_base / config->head_dim;
    const uint64_t                     attention_base = (uint64_t) args->row_base * width + args->k_base;
    const uint64_t                     gate_base =
        (uint64_t) args->row_base * 2U * width + first_head * 2U * config->head_dim + config->head_dim;

    *(volatile float *) (uintptr_t) constants                   = -1.4426950408889634f;
    *(volatile float *) (uintptr_t) (constants + sizeof(float)) = 1.0f;

    long checksum = 0;
    set_lsu_matrix<Constants::kLsuShared>(checksum, Constants::kTileM, Constants::kTileK * (int) sizeof(float),
                                          width * sizeof(float), Constants::kTileK * (int) sizeof(float));
    load_matrix<Constants::kLsuShared>((uint64_t) args->cols * sizeof(float), args->rows,
                                       Constants::kTileK * sizeof(float), Constants::kTileM, attention_uram,
                                       (uint64_t) (uintptr_t) (context->attention + attention_base));
    configure_gated_lsu<Constants::kLsuShared>(
        0, config->head_dim - 1, args->rows - 1, sizeof(uint16_t), sizeof(uint16_t), 2U * width * sizeof(uint16_t),
        sizeof(float), sizeof(float), Constants::kTileK * sizeof(float), 6, sizeof(uint16_t) - 1, sizeof(float) - 1);
    __tx_lsu_tcsr_write<Constants::kLsuShared * LSU_GROUPSZ + LSU_DIM2>(heads - 1);
    __tx_lsu_tcsr_write<Constants::kLsuShared * LSU_GROUPSZ + LSU_INDIM2STRIDE>(2U * config->head_dim *
                                                                                sizeof(uint16_t));
    __tx_lsu_tcsr_write<Constants::kLsuShared * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(config->head_dim * sizeof(float));
    __tx_lsu_dma_d2u<Constants::kLsuShared>(gate_uram, reinterpret_cast<uint64_t>(context->gate + gate_base));
    dayan_wait_lsu();

    const long chunks = args->cols / Constants::kVecLanes;
    issue_gated_lowp_to_f32<Format>(gate_uram, args->rows, chunks);
    dayan_wait_vec();
    ggml_dayan_vec_detail::issue_sigmoid_mul({ gate_uram, Constants::kTileK * (int) sizeof(float) },
                                             { attention_uram, Constants::kTileK * (int) sizeof(float) },
                                             { attention_uram, Constants::kTileK * (int) sizeof(float) },
                                             { constants, constants + sizeof(float), 0, 0, 0, 0, 0 }, args->rows,
                                             args->cols);
    dayan_wait_vec();
    convert_gated_f32_to_lowp<Format>(attention_uram, args->output, args->rows, args->cols);
    return 0;
}

template <typename Format>
static inline void run_gated_output_projection(const DayanGatedProjectionConfig * config,
                                               const float *                      attention,
                                               const uint16_t *                   gate,
                                               const uint16_t *                   weight,
                                               float *                            output,
                                               DayanMatmulPerf *                  perf) {
    DAYAN_PARAMS_IF(config == nullptr || attention == nullptr || gate == nullptr || weight == nullptr ||
                    output == nullptr || perf == nullptr || config->tokens == 0 || config->heads == 0 ||
                    config->head_dim == 0 || config->matmul.m != config->tokens ||
                    config->matmul.k != (uint64_t) config->heads * config->head_dim ||
                    config->head_dim > Constants::kTileK || config->head_dim % Constants::kVecLanes != 0 ||
                    Constants::kTileK % config->head_dim != 0 || config->matmul.m >= config->matmul.n) {
        DAYAN_PARAMS_ERROR();
    }

    const GatedInputContext input_context = { config, attention, gate };
    run_matmul_kernel<Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity, epilogue_identity,
                      epilogue_identity, epilogue_identity, epilogue_identity, load_gated_input<Format>,
                      kGatedInputScratchBytes>(&config->matmul, (const uint16_t *) attention, weight, nullptr, output,
                                               perf, &input_context);
}

}  // namespace ggml_dayan_matmul_detail

#endif
