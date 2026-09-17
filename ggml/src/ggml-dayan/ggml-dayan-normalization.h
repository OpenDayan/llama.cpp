#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct NormalizationLayout {
    enum : uint64_t {
        kTileElements = kDayanNativeTaskTileElements,
        kFloatBytes   = kTileElements * sizeof(float),
        kInput        = 0,
        kAux          = kInput + kFloatBytes,
        kValue        = kAux + kFloatBytes,
        kSquare       = kValue + kFloatBytes,
        kRms          = kSquare + kFloatBytes,
        kScaled       = kRms + kFloatBytes,
        kLowp         = kScaled + kFloatBytes,
        kScalar       = kLowp + kTileElements * sizeof(uint16_t),
        kVectorScalar = kScalar + 64,
        kVecReduce    = kVectorScalar + Constants::kVecBytes,
        kTotalBytes   = kVecReduce + kFloatBytes,
    };
};

static_assert(NormalizationLayout::kTotalBytes <= DAYAN_NATIVE_TASK_URAM_BYTES, "Native Task URAM layout is too large");
static_assert((NormalizationLayout::kScalar & 63U) == 0U, "Normalization scalar storage must be 64-byte aligned");

static inline bool dayan_native_task_mode_valid() {
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint32_t task  = dayan_task_id(DAYAN_TASK_X);
    const uint32_t mode  = dayan_task_mode();
    return (mode == 1U || (mode == 4U && tasks == 4U)) && tasks != 0U && task < tasks &&
           dayan_task_dim(DAYAN_TASK_Y) == 1U && dayan_task_dim(DAYAN_TASK_Z) == 1U;
}

static inline uint32_t dayan_native_task_id() {
    if (!dayan_native_task_mode_valid()) {
        DAYAN_PARAMS_ERROR();
    }
    return dayan_task_id(DAYAN_TASK_X);
}

static inline uint32_t dayan_native_task_count() {
    return dayan_task_dim(DAYAN_TASK_X);
}

static inline void configure_normalization_lsu() {
    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, NormalizationLayout::kFloatBytes, NormalizationLayout::kFloatBytes,
                      NormalizationLayout::kFloatBytes);
}

static inline bool normalization_negative_bits(uint32_t value) {
    return (value & 0x80000000U) != 0U && (value & 0x7fffffffU) != 0U;
}

static inline void normalization_pad_tail_bits(uint64_t uram, uint32_t elements, uint32_t value_bits) {
    const uint64_t mask = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kScalar;
    ggml_dayan_vec_detail::f32_pad_tail_bits(uram, mask, value_bits, elements);
}

static inline void normalization_pad_tail_address(uint64_t uram, uint32_t elements, uint64_t scalar) {
    const uint64_t mask = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kScalar;
    ggml_dayan_vec_detail::f32_pad_tail_address(uram, mask, scalar, elements);
}

static inline void normalization_load_bits(uint64_t uram,
                                           const float * dram,
                                           uint32_t      elements,
                                           uint32_t      padding_bits) {
    load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, uram,
                   (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
    normalization_pad_tail_bits(uram, elements, padding_bits);
}

static inline void normalization_load(uint64_t uram, const float * dram, uint32_t elements) {
    normalization_load_bits(uram, dram, elements, 0U);
}

using NormalizationLoad = void (*)(uint64_t, const void *, uint32_t);

static inline void normalization_load_f32(uint64_t uram, const void * dram, uint32_t elements) {
    normalization_load(uram, (const float *) dram, elements);
}

template <void (*Convert)()>
static inline void normalization_load_lowp(uint64_t uram, const void * dram, uint32_t elements) {
    const uint64_t staging = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kLowp;
    load_matrix<0>((uint64_t) elements * sizeof(uint16_t), 1,
                   NormalizationLayout::kTileElements * sizeof(uint16_t), 1, staging,
                   (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(uram, staging, elements);
    normalization_pad_tail_bits(uram, elements, 0U);
}

static inline void normalization_load_scalar_pad(uint64_t uram,
                                                 const float * dram,
                                                 uint32_t      elements,
                                                 uint64_t      scalar) {
    load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, uram,
                   (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
    normalization_pad_tail_address(uram, elements, scalar);
}

static inline void normalization_store_f32(float * dram, uint64_t uram, uint32_t elements) {
    store_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, uram,
                    (uint64_t) (uintptr_t) dram);
}

static inline void normalization_store_lowp(uint16_t * dram, uint64_t uram, uint32_t elements) {
    store_matrix<0>((uint64_t) elements * sizeof(uint16_t), 1, NormalizationLayout::kFloatBytes, 1, uram,
                    (uint64_t) (uintptr_t) dram);
}

static inline void normalization_vec_add(uint64_t left, uint64_t right, uint64_t output, uint32_t elements) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) left);
    __tx_vec_lsu_u2v<1, 1>((long) right);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

template <typename Format>
static inline void normalization_quantize(uint64_t output, uint64_t input, uint64_t scratch, uint32_t elements) {
    ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
        { input, (long) elements * (long) sizeof(float) },
        { scratch, (long) elements * (long) sizeof(float) },
        { output, (long) elements * (long) sizeof(uint16_t) }, 1, elements);
}

template <uint32_t InputType>
static inline void normalization_validate(const DayanNormConfig * config) {
    static_assert(InputType == kDayanDataTypeF32 || InputType == kDayanDataTypeBF16 ||
                  InputType == kDayanDataTypeF16);
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);

    if (config == nullptr || config->width == 0 || config->rows == 0 ||
        normalization_negative_bits(config->epsilon_bits) || config->input_ne[0] != config->width ||
        config->input_type != InputType || config->input_nb[0] != input_bytes || config->input_ne[1] == 0 ||
        config->input_ne[2] == 0 || config->input_ne[3] == 0 ||
        (uint64_t) config->input_ne[1] * config->input_ne[2] * config->input_ne[3] != config->rows) {
        DAYAN_PARAMS_ERROR();
    }
    (void) dayan_native_task_id();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_thread_exit(-12);
    }
}

static inline const void * normalization_input_row(const DayanNormConfig * config, const void * input, uint32_t row) {
    uint32_t       index  = row;
    const uint32_t index1 = index % config->input_ne[1];
    index /= config->input_ne[1];
    const uint32_t index2 = index % config->input_ne[2];
    const uint32_t index3 = index / config->input_ne[2];
    return (const char *) input + (uint64_t) index1 * config->input_nb[1] +
           (uint64_t) index2 * config->input_nb[2] + (uint64_t) index3 * config->input_nb[3];
}

static inline void normalization_start_loop(long prologue, long loop_end, long instructions) {
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<ggml_dayan_vec_detail::kVecSync>(0);
    __tx_vec_tcsr_write<ggml_dayan_vec_detail::kVecControl>(
        prologue | (prologue << 8) | (loop_end << 16) | (instructions << 24) | (instructions << 32));
}

template <bool Square, bool Centered = false>
static inline void normalization_accumulate(uint64_t accumulator, uint64_t input, uint64_t mean,
                                            uint32_t elements, bool initialize) {
    constexpr long loop_end = 3 + (Square ? 1 : 0) + (Centered ? 2 : 0);
    const long instructions = loop_end + 14 + (initialize ? 0 : 2);

    ggml_dayan_vec_detail::configure_vec_grid(1);
    __tx_vec_tcsr_write<ggml_dayan_vec_detail::kVecDim1>(ggml_dayan_vec_detail::f32_chunks(elements) - 1);
    configure_vec_iterator<kVecV0Base>(0, Constants::kVecBytes, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    // Q1 holds lane sums across DIM1; Q0 holds the current input.
    __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    if constexpr (Centered) {
        __tx_vec_lsu_u2v_broadcast32<0, 1>((long) mean);
        __tx_vec_vv_fp32_sub<0, 1, 0, 0>();
    }
    if constexpr (Square) {
        __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    }
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(0);
    ggml_dayan_vec_detail::emit_f32_sum_stages();
    if (!initialize) {
        __tx_vec_lsu_u2v_broadcast32<1, 1>((long) accumulator);
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>((long) accumulator);
    normalization_start_loop(1, loop_end, instructions);
    dayan_wait_vec();
}

static inline void normalization_sum_accumulate(uint64_t accumulator, uint64_t input, uint32_t elements, bool initialize) {
    normalization_accumulate<false>(accumulator, input, 0, elements, initialize);
}

static inline void normalization_square_sum_accumulate(uint64_t accumulator, uint64_t input, uint32_t elements, bool initialize) {
    normalization_accumulate<true>(accumulator, input, 0, elements, initialize);
}

static inline void normalization_centered_square_sum_accumulate(uint64_t accumulator, uint64_t input, uint64_t mean,
                                                                 uint32_t elements, bool initialize) {
    normalization_accumulate<true, true>(accumulator, input, mean, elements, initialize);
}

static inline void normalization_mean(uint64_t output, uint64_t sum, uint32_t inv_width_bits) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) sum);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) inv_width_bits);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_inverse_rms_finalize(uint64_t inverse, uint32_t inv_width_bits, uint32_t epsilon_bits) {
    constexpr long instructions = 7;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) inverse);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) inv_width_bits);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) epsilon_bits);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 4>();
    __tx_vec_lsu_v2u<0, 3>((long) inverse);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_apply_rms(uint64_t output, uint64_t input, uint64_t inverse, uint32_t elements) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) inverse);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_apply_rms_scale(uint64_t rms,
                                                 uint64_t scaled,
                                                 uint64_t input,
                                                 uint64_t inverse,
                                                 uint64_t scale,
                                                 uint32_t elements,
                                                 bool write_rms) {
    const long instructions = write_rms ? 8 : 6;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v_broadcast32<1, 2>((long) inverse);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    if (write_rms) {
        __tx_vec_shuffle_xor<1, 0>(0);
    }
    __tx_vec_lsu_u2v<1, 1>((long) scale);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) scaled);
    if (write_rms) {
        __tx_vec_lsu_v2u<17, 3>((long) rms);
    }
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_center_scale(uint64_t output,
                                              uint64_t input,
                                              uint64_t mean,
                                              uint64_t inverse,
                                              uint32_t elements) {
    constexpr long instructions = 6;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) mean);
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) inverse);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

template <typename Format, void (*Convert)(), uint32_t InputType>
static inline void normalization_cached_reduce(uint64_t values, uint64_t right, uint64_t mean, uint64_t result,
                                               uint32_t width, uint32_t rows, uint32_t inv_width_bits,
                                               uint32_t epsilon_bits, bool square, bool centered, bool residual) {
    const long loop_end = 3 + (square ? 1 : 0) + (centered ? 3 : 0) +
                          (residual ? 3 + (InputType == kDayanDataTypeF32 ? 0 : 2) : 0);
    const long instructions = loop_end + 16 + (square ? 3 : 0);
    const long row_bytes = (long) width * sizeof(float);
    ggml_dayan_vec_detail::configure_vec_grid(rows);
    __tx_vec_tcsr_write<ggml_dayan_vec_detail::kVecDim1>(width / Constants::kVecLanes - 1);
    configure_vec_iterator<kVecV0Base>(0, Constants::kVecBytes, row_bytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, Constants::kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, Constants::kVecBytes, row_bytes, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
    __tx_vec_lsu_u2v<0, 0>((long) values);
    if (residual) {
        __tx_vec_lsu_u2v<0, 2>((long) right);
        __tx_vec_vv_fp32_add<0, 0, 1, 0>();
        if constexpr (InputType != kDayanDataTypeF32) {
            if constexpr (Format::kMatDataType == BaseNPUInstBuilder::DT_BF16) {
                __tx_vec_cvt_bf16_fp32<0, 0, 0>();
            } else {
                __tx_vec_cvt_fp16_fp32<0, 0, 0>();
            }
            Convert();
        }
        __tx_vec_lsu_v2u<0, 0>((long) values);
    }
    if (centered) {
        __tx_vec_lsu_u2v_broadcast32<0, 1>((long) mean);
        __tx_vec_vv_fp32_sub<0, 1, 0, 0>();
        __tx_vec_lsu_v2u<0, 0>((long) values);
    }
    if (square) {
        __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    }
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
    __tx_vec_shuffle_xor<0, 16>(0);
    ggml_dayan_vec_detail::emit_f32_sum_stages();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) inv_width_bits);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    if (square) {
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) epsilon_bits);
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
        __tx_vec_v_fp32_sfu<0, 0, 4>();
    }
    __tx_vec_lsu_v2u<0, 1>((long) result);
    normalization_start_loop(1, loop_end, instructions);
    dayan_wait_vec();
}

template <uint32_t InputType, void (*Convert)()>
static inline void normalization_cached_load(const DayanNormConfig * config, const void * source,
                                             uint32_t row, uint32_t rows, uint64_t values, uint64_t packed) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t bytes = (uint64_t) config->width * input_bytes;
    long checksum = 0;
    set_lsu_matrix<0>(checksum, rows, bytes, config->input_nb[1], bytes);
    load_matrix<0>(bytes, rows, bytes, rows, InputType == kDayanDataTypeF32 ? values : packed,
                   (uint64_t) (uintptr_t) normalization_input_row(config, source, row));
    dayan_wait_lsu();
    if constexpr (InputType != kDayanDataTypeF32) {
        ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(values, packed, config->width, rows, bytes,
                                                         (long) config->width * sizeof(float));
    }
}

static inline void normalization_cached_store(void * destination, uint64_t source, uint32_t width, uint32_t rows, bool packed) {
    long checksum = 0;
    if (packed) {
        set_lsu_matrix<0>(checksum, width / Constants::kVecLanes, Constants::kVecBytes / 2,
                          Constants::kVecBytes, Constants::kVecBytes / 2);
        __tx_lsu_tcsr_write<LSU_DIM2>(rows - 1);
        __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>((long) width * sizeof(float));
        __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>((long) width * sizeof(uint16_t));
    } else {
        set_lsu_matrix<0>(checksum, rows, (long) width * sizeof(float),
                          (long) width * sizeof(float), (long) width * sizeof(float));
    }
    __tx_lsu_dma_u2d<0>((long) (uintptr_t) destination, (long) source);
    dayan_wait_lsu();
}

template <typename Format>
static inline void normalization_cached_pack(uint64_t output, uint64_t input, uint32_t width, uint32_t rows) {
    ggml_dayan_vec_detail::configure_vec_grid(width / Constants::kVecLanes, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, (long) width * sizeof(float));
    begin_vec_block(4);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    if constexpr (Format::kMatDataType == BaseNPUInstBuilder::DT_BF16) {
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
    } else {
        __tx_vec_cvt_fp16_fp32<0, 0, 0>();
    }
    __tx_vec_shuffle_compress<0, 0>(0);
    __tx_vec_lsu_v2u<0, 0>((long) output);
    start_vec_block(4);
    dayan_wait_vec();
}

template <typename Format>
static inline void normalization_cached_apply(uint64_t values, uint64_t output, uint64_t scale, uint64_t inverse,
                                              uint32_t width, uint32_t rows, bool write_rms, bool pack) {
    const long instructions = 4 + (scale != 0 ? 2 : 0) + (write_rms ? 1 : 0) + (pack ? 2 : 0);
    ggml_dayan_vec_detail::configure_vec_grid(rows);
    __tx_vec_tcsr_write<ggml_dayan_vec_detail::kVecDim1>(width / Constants::kVecLanes - 1);
    configure_vec_iterator<kVecV0Base>(0, Constants::kVecBytes, (long) width * sizeof(float), 0);
    configure_vec_iterator<kVecV1Base>(0, 0, Constants::kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, Constants::kVecBytes, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) inverse);
    __tx_vec_lsu_u2v<0, 0>((long) values);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    if (write_rms) {
        __tx_vec_lsu_v2u<0, 0>((long) values);
    }
    if (scale != 0) {
        __tx_vec_lsu_u2v<0, 2>((long) scale);
        __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    }
    if (pack) {
        if constexpr (Format::kMatDataType == BaseNPUInstBuilder::DT_BF16) {
            __tx_vec_cvt_bf16_fp32<0, 0, 0>();
        } else {
            __tx_vec_cvt_fp16_fp32<0, 0, 0>();
        }
        __tx_vec_shuffle_compress<0, 0>(0);
    }
    __tx_vec_lsu_v2u<0, 0>((long) output);
    normalization_start_loop(1, instructions, instructions);
    dayan_wait_vec();
}

template <typename Format, void (*Convert)(), uint32_t InputType>
static bool run_normalization_cached(const DayanNormConfig * config, const void * left, const void * right,
                                     const float * scale, void * residual, float * rms, float * scaled, uint16_t * lowp,
                                     uint32_t flags, bool centered) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t row_bytes = (uint64_t) config->width * sizeof(float);
    const uint64_t scale_bytes = scale != nullptr ? row_bytes : 0;
    const uint64_t packed_bytes = InputType == kDayanDataTypeF32 ? 0 : row_bytes / 2;
    const uint64_t bytes_per_row = 2 * row_bytes + packed_bytes + 2 * Constants::kVecBytes;
    const uint32_t alignment = InputType == kDayanDataTypeF32 ? Constants::kVecLanes : 2 * Constants::kVecLanes;
    const uint64_t capacity = dayan_uram_size() < DAYAN_NORM_TASK_URAM_BYTES ? dayan_uram_size() : DAYAN_NORM_TASK_URAM_BYTES;
    if (config->width % alignment != 0 || scale_bytes + bytes_per_row > capacity) {
        return false;
    }
    const uint32_t batch_rows = (uint32_t) ((capacity - scale_bytes) / bytes_per_row);
    const uint64_t values = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t work = values + batch_rows * row_bytes;
    const uint64_t packed = work + batch_rows * row_bytes;
    const uint64_t mean = packed + batch_rows * packed_bytes;
    const uint64_t inverse = mean + batch_rows * Constants::kVecBytes;
    const uint64_t scale_uram = inverse + batch_rows * Constants::kVecBytes;
    const bool has_residual = (flags & kDayanNormHasResidual) != 0;
    const bool write_residual = (flags & kDayanNormWriteResidual) != 0;
    const bool write_rms = (flags & kDayanNormWriteRms) != 0;
    const bool write_scaled = (flags & kDayanNormWriteScaled) != 0;
    const bool write_lowp = (flags & kDayanNormWriteLowp) != 0;
    if (scale != nullptr) {
        long checksum = 0;
        set_lsu_matrix<0>(checksum, 1, row_bytes, row_bytes, row_bytes);
        __tx_lsu_dma_d2u<0>((long) scale_uram, (long) (uintptr_t) scale);
        dayan_wait_lsu();
    }
    const uint32_t task = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    const uint32_t end = (uint32_t) ((uint64_t) config->rows * (task + 1) / tasks);
    for (uint32_t row = (uint32_t) ((uint64_t) config->rows * task / tasks); row < end;) {
        uint32_t rows = min_int((int) (end - row), (int) batch_rows);
        rows = min_int((int) rows, (int) (config->input_ne[1] - row % config->input_ne[1]));
        normalization_cached_load<InputType, Convert>(config, left, row, rows, values, packed);
        if (has_residual) {
            normalization_cached_load<InputType, Convert>(config, right, row, rows, work, packed);
        }
        if (centered) {
            normalization_cached_reduce<Format, Convert, InputType>(values, work, 0, mean, config->width, rows,
                config->inv_width_bits, config->epsilon_bits, false, false, false);
        }
        normalization_cached_reduce<Format, Convert, InputType>(values, work, mean, inverse, config->width, rows,
            config->inv_width_bits, config->epsilon_bits, true, centered, has_residual);
        const uint64_t offset = (uint64_t) row * config->width;
        if (write_residual) {
            if constexpr (InputType == kDayanDataTypeF32) {
                normalization_cached_store((char *) residual + offset * input_bytes, values, config->width, rows, false);
            } else {
                normalization_cached_pack<Format>(work, values, config->width, rows);
                normalization_cached_store((char *) residual + offset * input_bytes, work, config->width, rows, true);
            }
        }
        normalization_cached_apply<Format>(values, work, scale != nullptr ? scale_uram : 0, inverse,
            config->width, rows, write_rms, write_lowp && !write_scaled);
        if (write_rms) {
            normalization_cached_store(rms + offset, values, config->width, rows, false);
        }
        if (write_scaled) {
            normalization_cached_store(scaled + offset, work, config->width, rows, false);
        }
        if (write_lowp) {
            if (write_scaled) {
                normalization_cached_pack<Format>(work, work, config->width, rows);
            }
            normalization_cached_store(lowp + offset, work, config->width, rows, true);
        }
        row += rows;
    }
    return true;
}

template <typename Format, void (*Convert)(), uint32_t InputType, NormalizationLoad Load, bool HasResidual>
static uint64_t normalization_inverse_rms(const DayanNormConfig * config,
                                          const void *            left,
                                          const void *            right,
                                          void *                  residual,
                                          bool                    write_residual) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t uram    = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input   = uram + NormalizationLayout::kInput;
    const uint64_t aux     = uram + NormalizationLayout::kAux;
    const uint64_t value   = uram + NormalizationLayout::kValue;
    const uint64_t inverse = uram + NormalizationLayout::kVectorScalar;

    for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
        const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
        if constexpr (HasResidual) {
            Load(input, (const char *) left + (uint64_t) offset * input_bytes, elements);
            Load(aux, (const char *) right + (uint64_t) offset * input_bytes, elements);
            normalization_vec_add(input, aux, value, elements);
            if constexpr (InputType != kDayanDataTypeF32) {
                const uint64_t lowp = uram + NormalizationLayout::kLowp;
                normalization_quantize<Format>(lowp, value, input, elements);
                configure_normalization_lsu();
                ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(value, lowp, elements);
                if (write_residual) {
                    normalization_store_lowp((uint16_t *) residual + offset, lowp, elements);
                    dayan_wait_lsu();
                }
            } else if (write_residual) {
                    normalization_store_f32((float *) residual + offset, value, elements);
                dayan_wait_lsu();
            }
        } else {
            Load(value, (const char *) left + (uint64_t) offset * input_bytes, elements);
        }

        normalization_square_sum_accumulate(inverse, value, elements, offset == 0);
    }

    normalization_inverse_rms_finalize(inverse, config->inv_width_bits, config->epsilon_bits);
    return inverse;
}

template <typename Format, void (*Convert)(), uint32_t InputType, NormalizationLoad Load, bool HasResidual, bool Quantize>
static void run_rms_norm_scale(const DayanNormPipelineConfig * config,
                               const void *                    left,
                               const void *                    right,
                               const float *                   scale,
                               void *                          residual,
                               float *                         rms,
                               float *                         scaled,
                               uint16_t *                      lowp) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const bool write_residual = (config->flags & kDayanNormWriteResidual) != 0U;
    const bool write_rms      = (config->flags & kDayanNormWriteRms) != 0U;
    const bool write_scaled   = (config->flags & kDayanNormWriteScaled) != 0U;

    const DayanNormConfig * norm = &config->norm;
    normalization_validate<InputType>(norm);
    if (left == nullptr || scale == nullptr || (HasResidual && right == nullptr) ||
        (write_residual && residual == nullptr) || (write_rms && rms == nullptr) ||
        (write_scaled && scaled == nullptr) || (Quantize && lowp == nullptr)) {
        DAYAN_PARAMS_ERROR();
    }

    if (run_normalization_cached<Format, Convert, InputType>(norm, left, right, scale, residual, rms, scaled, lowp,
                                                             config->flags, false)) {
        return;
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    for (uint32_t row = dayan_native_task_id(); row < norm->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset   = (uint64_t) row * norm->width;
        const void *    left_row     = normalization_input_row(norm, left, row);
        const void *    right_row    = HasResidual ? normalization_input_row(norm, right, row) : nullptr;
        void * residual_row = HasResidual && write_residual ?
                                  (char *) residual + row_offset * input_bytes :
                                  nullptr;
        const uint64_t inverse_uram = normalization_inverse_rms<Format, Convert, InputType, Load, HasResidual>(
            norm, left_row, right_row, residual_row, write_residual);
        const uint64_t uram         = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t input        = uram + NormalizationLayout::kInput;
        const uint64_t value        = uram + NormalizationLayout::kValue;
        const uint64_t aux          = uram + NormalizationLayout::kAux;
        const uint64_t rms_uram     = uram + NormalizationLayout::kRms;
        const uint64_t scaled_uram  = uram + NormalizationLayout::kScaled;
        const uint64_t lowp_uram    = uram + NormalizationLayout::kLowp;
        for (uint32_t offset = 0; offset < norm->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (norm->width - offset), (int) NormalizationLayout::kTileElements);
            if constexpr (HasResidual) {
                if (write_residual) {
                    Load(value, (const char *) residual_row + (uint64_t) offset * input_bytes, elements);
                } else {
                    Load(input, (const char *) left_row + (uint64_t) offset * input_bytes, elements);
                    Load(aux, (const char *) right_row + (uint64_t) offset * input_bytes, elements);
                    normalization_vec_add(input, aux, value, elements);
                    if constexpr (InputType != kDayanDataTypeF32) {
                        normalization_quantize<Format>(lowp_uram, value, input, elements);
                        configure_normalization_lsu();
                        ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(value, lowp_uram, elements);
                    }
                }
            } else {
                Load(value, (const char *) left_row + (uint64_t) offset * input_bytes, elements);
            }
            normalization_load(aux, scale + offset, elements);

            normalization_apply_rms_scale(
                rms_uram, scaled_uram, value, inverse_uram, aux, elements, write_rms);
            if constexpr (Quantize) {
                normalization_quantize<Format>(lowp_uram, scaled_uram, input, elements);
                configure_normalization_lsu();
            }

            bool issued_store = false;
            if (write_rms) {
                normalization_store_f32(rms + row_offset + offset, rms_uram, elements);
                issued_store = true;
            }
            if (write_scaled) {
                normalization_store_f32(scaled + row_offset + offset, scaled_uram, elements);
                issued_store = true;
            }
            if constexpr (Quantize) {
                normalization_store_lowp(lowp + row_offset + offset, lowp_uram, elements);
                issued_store = true;
            }
            if (issued_store) {
                dayan_wait_lsu();
            }
        }
    }
}

template <uint32_t InputType, NormalizationLoad Load>
static inline void run_rms_norm_typed(const DayanNormConfig * config, const void * input, float * output) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    normalization_validate<InputType>(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    if constexpr (InputType == kDayanDataTypeF16) {
        if (run_normalization_cached<Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, InputType>(
                config, input, nullptr, nullptr, nullptr, nullptr, output, nullptr, kDayanNormWriteScaled, false)) {
            return;
        }
    } else {
        if (run_normalization_cached<Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, InputType>(
                config, input, nullptr, nullptr, nullptr, nullptr, output, nullptr, kDayanNormWriteScaled, false)) {
            return;
        }
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset   = (uint64_t) row * config->width;
        const void *    input_row    = normalization_input_row(config, input, row);
        const uint64_t inverse_uram = normalization_inverse_rms<
            Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, InputType, Load, false>(
            config, input_row, nullptr, nullptr, false);
        const uint64_t uram         = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t value        = uram + NormalizationLayout::kValue;
        const uint64_t rms_uram     = uram + NormalizationLayout::kRms;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            Load(value, (const char *) input_row + (uint64_t) offset * input_bytes, elements);
            normalization_apply_rms(rms_uram, value, inverse_uram, elements);
            normalization_store_f32(output + row_offset + offset, rms_uram, elements);
            dayan_wait_lsu();
        }
    }
}

static inline void run_rms_norm(const DayanNormConfig * config, const void * input, float * output) {
    if (config->input_type == kDayanDataTypeF32) {
        run_rms_norm_typed<kDayanDataTypeF32, normalization_load_f32>(config, input, output);
    } else if (config->input_type == kDayanDataTypeBF16) {
        run_rms_norm_typed<kDayanDataTypeBF16,
                           normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>>(config, input, output);
    } else if (config->input_type == kDayanDataTypeF16) {
        run_rms_norm_typed<kDayanDataTypeF16,
                           normalization_load_lowp<__tx_vec_cvt_fp32_fp16<0, 0, 0>>>(config, input, output);
    } else {
        DAYAN_PARAMS_ERROR();
    }
}

static inline void normalization_centered(const float * input,
                                          float *       output,
                                          uint64_t      elements,
                                          uint32_t      epsilon_bits,
                                          uint32_t      inv_width_bits) {
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t mean       = uram + NormalizationLayout::kRms;
    const uint64_t scaled     = uram + NormalizationLayout::kScaled;
    const uint64_t inverse    = uram + NormalizationLayout::kVectorScalar;

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        normalization_sum_accumulate(inverse, input_uram, count, offset == 0);
    }

    normalization_mean(mean, inverse, inv_width_bits);
    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load_scalar_pad(input_uram, input + offset, count, mean);
        normalization_centered_square_sum_accumulate(inverse, input_uram, mean, count, offset == 0);
    }

    normalization_inverse_rms_finalize(inverse, inv_width_bits, epsilon_bits);

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        normalization_center_scale(scaled, input_uram, mean, inverse, count);
        normalization_store_f32(output + offset, scaled, count);
        dayan_wait_lsu();
    }
}

static inline void run_norm(const DayanNormConfig * config, const float * input, float * output) {
    normalization_validate<kDayanDataTypeF32>(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    if (run_normalization_cached<Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeF32>(
            config, input, nullptr, nullptr, nullptr, nullptr, output, nullptr, kDayanNormWriteScaled, true)) {
        return;
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t offset = (uint64_t) row * config->width;
        normalization_centered(
            (const float *) normalization_input_row(config, input, row), output + offset, config->width,
            config->epsilon_bits, config->inv_width_bits);
    }
}

static inline void run_group_norm(const DayanGroupNormConfig * config, const float * input, float * output) {
    if (config == nullptr || config->width == 0 || config->height == 0 || config->channels == 0 ||
        config->batches == 0 || config->groups == 0 || config->groups > config->channels || input == nullptr ||
        output == nullptr || normalization_negative_bits(config->epsilon_bits)) {
        DAYAN_PARAMS_ERROR();
    }
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_thread_exit(-12);
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    const uint32_t channels_per_group = (config->channels + config->groups - 1U) / config->groups;
    const uint64_t channel_elements   = (uint64_t) config->width * config->height;
    const uint64_t work_items         = (uint64_t) config->batches * config->groups;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint32_t batch         = (uint32_t) (work / config->groups);
        const uint32_t group         = (uint32_t) (work % config->groups);
        const uint32_t channel_start = group * channels_per_group;
        const uint32_t channel_end   = channel_start + channels_per_group < config->channels ?
                                           channel_start + channels_per_group :
                                           config->channels;
        const uint64_t offset        = ((uint64_t) batch * config->channels + channel_start) * channel_elements;
        const uint64_t elements      = (uint64_t) (channel_end - channel_start) * channel_elements;
        normalization_centered(input + offset, output + offset, elements, config->epsilon_bits,
            channel_end - channel_start == channels_per_group ? config->inv_group_bits : config->inv_last_group_bits);
    }
}

template <typename Format, uint32_t InputType, NormalizationLoad Load>
static inline void run_residual_add(const DayanElementwiseConfig * config,
                                    const void *                   left,
                                    const void *                   right,
                                    void *                         output) {
    if (config == nullptr || config->elements == 0 || left == nullptr || right == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_thread_exit(-12);
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    const uint64_t uram  = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input = uram + NormalizationLayout::kInput;
    const uint64_t aux   = uram + NormalizationLayout::kAux;
    const uint64_t value = uram + NormalizationLayout::kValue;
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        Load(input, (const char *) left + offset * input_bytes, elements);
        Load(aux, (const char *) right + offset * input_bytes, elements);
        normalization_vec_add(input, aux, value, elements);
        if constexpr (InputType == kDayanDataTypeF32) {
            normalization_store_f32((float *) output + offset, value, elements);
        } else {
            const uint64_t lowp = uram + NormalizationLayout::kLowp;
            normalization_quantize<Format>(lowp, value, input, elements);
            configure_normalization_lsu();
            normalization_store_lowp((uint16_t *) output + offset, lowp, elements);
        }
        dayan_wait_lsu();
    }
}

template <typename Format, void (*Convert)(), uint32_t InputType, NormalizationLoad Load, bool HasResidual>
static inline void run_rms_norm_pipeline_typed(const DayanNormPipelineConfig * config,
                                               const void *                    left,
                                               const void *                    right,
                                               const float *                   scale,
                                               void *                          residual,
                                               float *                         rms,
                                               float *                         scaled,
                                               uint16_t *                      lowp) {
    const bool write_lowp = (config->flags & kDayanNormWriteLowp) != 0U;
    if (!write_lowp) {
        run_rms_norm_scale<Format, Convert, InputType, Load, HasResidual, false>(
            config, left, right, scale, residual, rms, scaled, lowp);
    } else {
        run_rms_norm_scale<Format, Convert, InputType, Load, HasResidual, true>(
            config, left, right, scale, residual, rms, scaled, lowp);
    }
}

static inline void run_rms_norm_pipeline(const DayanNormPipelineConfig * config,
                                         const void *                    left,
                                         const void *                    right,
                                         const float *                   scale,
                                         void *                          residual,
                                         float *                         rms,
                                         float *                         scaled,
                                         uint16_t *                      lowp) {
    const uint32_t write_flags = kDayanNormWriteResidual | kDayanNormWriteRms | kDayanNormWriteScaled |
                                 kDayanNormWriteLowp;
    const uint32_t allowed = kDayanNormHasResidual | write_flags;
    if (config == nullptr || (config->flags & write_flags) == 0U || (config->flags & ~allowed) != 0U ||
        ((config->flags & kDayanNormWriteResidual) != 0U && (config->flags & kDayanNormHasResidual) == 0U)) {
        DAYAN_PARAMS_ERROR();
    }

    const bool has_residual = (config->flags & kDayanNormHasResidual) != 0U;
    const bool write_lowp   = (config->flags & kDayanNormWriteLowp) != 0U;
    if ((write_lowp && config->output_type != kDayanDataTypeBF16 && config->output_type != kDayanDataTypeF16) ||
        (has_residual && write_lowp && config->norm.input_type != kDayanDataTypeF32 &&
         config->norm.input_type != config->output_type)) {
        DAYAN_PARAMS_ERROR();
    }

    if (config->norm.input_type == kDayanDataTypeF32) {
        if (write_lowp && config->output_type == kDayanDataTypeF16) {
            if (has_residual) {
                run_rms_norm_pipeline_typed<
                    Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, true>(
                    config, left, right, scale, residual, rms, scaled, lowp);
            } else {
                run_rms_norm_pipeline_typed<
                    Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, false>(
                    config, left, right, scale, residual, rms, scaled, lowp);
            }
        } else if (has_residual) {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, true>(
                config, left, right, scale, residual, rms, scaled, lowp);
        } else {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, false>(
                config, left, right, scale, residual, rms, scaled, lowp);
        }
    } else if (config->norm.input_type == kDayanDataTypeBF16) {
        if (has_residual) {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeBF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>, true>(
                config, left, right, scale, residual, rms, scaled, lowp);
        } else {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeBF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>, false>(
                config, left, right, scale, residual, rms, scaled, lowp);
        }
    } else if (config->norm.input_type == kDayanDataTypeF16) {
        if (has_residual) {
            run_rms_norm_pipeline_typed<
                Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_fp16<0, 0, 0>>, true>(
                config, left, right, scale, residual, rms, scaled, lowp);
        } else {
            run_rms_norm_pipeline_typed<
                Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_fp16<0, 0, 0>>, false>(
                config, left, right, scale, residual, rms, scaled, lowp);
        }
    } else {
        DAYAN_PARAMS_ERROR();
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif
