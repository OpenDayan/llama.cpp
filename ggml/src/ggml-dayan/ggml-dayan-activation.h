#pragma once

#if defined(__DAYAN_DEVICE__)

#    include <dayan_intrinsics.h>
#    include <dayan_sys.h>
#    include <stdint.h>

namespace ggml_dayan_vec_detail {

enum VecTcsr {
    kVecControl = 0,
    kVecSync    = 1,
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

enum : long {
    kVecLanes = 64,
    kVecBytes = kVecLanes * sizeof(float),
};

struct F32Tensor {
    uint64_t address;
    long     row_stride;
};

struct LowpTensor {
    uint64_t address;
    long     row_stride;
};

struct ActivationConstants {
    uint64_t neg_log2e;
    uint64_t one;
    uint64_t gelu_a;
    uint64_t sqrt_2_over_pi;
    uint64_t neg_2_log2e;
    uint64_t two;
    uint64_t half;
};

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

static inline void configure_vec_grid(long dim2, long dim3 = 1) {
    __tx_vec_tcsr_write<kVecDim0>(0);
    __tx_vec_tcsr_write<kVecDim1>(0);
    __tx_vec_tcsr_write<kVecDim2>(dim2 - 1);
    __tx_vec_tcsr_write<kVecDim3>(dim3 - 1);
    __tx_vec_tcsr_write<kVecDim4>(0);
    __tx_vec_tcsr_write<kVecDim5>(0);
}

static inline long vec_control(long instructions) {
    const uint64_t count = (uint64_t) instructions & 0xffU;
    return (long) (count | (count << 8) | (count << 16) | (count << 24) | (count << 32));
}

static inline void begin_vec_block(long instructions) {
    long free_entries;

    do {
        __tx_vec_tcsr_read<kVecControl>(free_entries);
    } while (free_entries < instructions);
}

static inline void start_vec_block(long instructions) {
    // Keep CPU URAM stores before VEC submission.
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecSync>(0);
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

static inline uint32_t f32_bits(float value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

static inline long f32_chunks(long elements) {
    return (elements + kVecLanes - 1) / kVecLanes;
}

static inline void configure_f32_iterators() {
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, 0);
}

static inline void configure_f32_tile(long elements, long input1_stride = kVecBytes) {
    configure_vec_grid(f32_chunks(elements));
    __tx_vec_tcsr_write<kVecV1Base + 3>(input1_stride);
}

template <uint64_t LaneMask, long XorMask>
static inline void swap_vec_lane_bits() {
    __tx_vec_lsu_g2s<0>((long) LaneMask);
    __tx_vec_shuffle_xor<1, 0>(XorMask);
    __tx_vec_vv_mux2<0, 0, 16, 0>();
}

template <void (*Convert)()>
static inline void issue_lowp_to_f32(uint64_t output, uint64_t input, long elements,
                                    long rows = 1, long input_row_stride = 0, long output_row_stride = 0) {
    constexpr long instructions = 28;
    const long     blocks       = (elements + 2 * kVecLanes - 1) / (2 * kVecLanes);

    configure_vec_grid(blocks, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, input_row_stride);
    configure_vec_iterator<kVecV3Base>(0, 0, 2 * kVecBytes, output_row_stride);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    swap_vec_lane_bits<0x0000ffffffff0000ULL, 48>();
    swap_vec_lane_bits<0x00ffff0000ffff00ULL, 24>();
    swap_vec_lane_bits<0x0ff00ff00ff00ff0ULL, 12>();
    swap_vec_lane_bits<0x3c3c3c3c3c3c3c3cULL, 6>();
    swap_vec_lane_bits<0x6666666666666666ULL, 3>();
    __tx_vec_shuffle_xor<1, 0>(1);
    __tx_vec_lsu_g2s<0>((long) 0xaaaaaaaaaaaaaaaaULL);

    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_v_int32_srl<1, 0, 16>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    Convert();
    __tx_vec_lsu_v2u<0, 3>((long) output);

    __tx_vec_vv_mux2<0, 17, 3, 0>();
    __tx_vec_v_int32_srl<1, 0, 16>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    Convert();
    __tx_vec_lsu_v2u<0, 3>((long) (output + kVecBytes));
    start_vec_block(instructions);
    dayan_wait_vec();
    __tx_vec_tcsr_write<kVecV3Base + 3>(kVecBytes);
}

static inline uint64_t f32_tail_mask(long elements) {
    const long tail = elements % kVecLanes;
    return tail == 0 ? 0 : ~((uint64_t{ 1 } << tail) - 1U);
}

static inline void f32_pad_tail_bits(uint64_t output, uint64_t mask_address, uint32_t value_bits, long elements) {
    const uint64_t mask = f32_tail_mask(elements);
    if (mask == 0) {
        return;
    }

    *reinterpret_cast<volatile uint64_t *>(mask_address) = mask;

    constexpr long instructions = 3;
    configure_vec_grid(1);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2s<0, 0>((long) mask_address);
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) value_bits);
    __tx_vec_lsu_v2u_mask<0, 0, 0>((long) (output + (uint64_t) (elements / kVecLanes) * kVecBytes));
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_pad_tail_address(uint64_t output, uint64_t mask_address, uint64_t scalar, long elements) {
    const uint64_t mask = f32_tail_mask(elements);
    if (mask == 0) {
        return;
    }

    *reinterpret_cast<volatile uint64_t *>(mask_address) = mask;

    constexpr long instructions = 3;
    configure_vec_grid(1);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2s<0, 0>((long) mask_address);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) scalar);
    __tx_vec_lsu_v2u_mask<0, 0, 0>((long) (output + (uint64_t) (elements / kVecLanes) * kVecBytes));
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void emit_f32_sum_stages() {
    __tx_vec_shuffle_xor<1, 0>(32);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
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

static inline void emit_f32_max_stages() {
    __tx_vec_shuffle_xor<1, 0>(32);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
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

static inline void configure_activation(const F32Tensor &  input0,
                                        const F32Tensor &  input1,
                                        const F32Tensor &  output,
                                        long               rows,
                                        long               columns) {
    const long chunks = (columns + kVecLanes - 1) / kVecLanes;

    configure_vec_grid(chunks, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, input0.row_stride);
    configure_vec_iterator<kVecV1Base>(0, 0, kVecBytes, input1.row_stride);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, output.row_stride);
}

template <long Group>
static inline void copy_packed_lowp(const F32Tensor & scratch, const LowpTensor & output, long rows, long chunks) {
    long checksum = 0;
    set_lsu_matrix<Group>(checksum, chunks, kVecLanes * sizeof(uint16_t), kVecBytes, kVecLanes * sizeof(uint16_t));
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(rows - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(scratch.row_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(output.row_stride);
    __tx_lsu_dma_u2u<Group>(output.address, scratch.address);
    dayan_wait_lsu();
}

template <long MatDataType, long LsuGroup>
static inline void convert_f32_to_lowp(const F32Tensor &  input,
                                       const F32Tensor &  scratch,
                                       const LowpTensor & output,
                                       long               rows,
                                       long               columns) {
    enum : long {
        kInstructions = 4,
    };

    const long chunks = (columns + kVecLanes - 1) / kVecLanes;
    configure_vec_grid(chunks, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, input.row_stride);
    configure_vec_iterator<kVecV1Base>(0, 0, kVecBytes, scratch.row_stride);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, scratch.row_stride);
    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(input.address);
    if constexpr (MatDataType == BaseNPUInstBuilder::DT_BF16) {
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
    } else {
        static_assert(MatDataType == BaseNPUInstBuilder::DT_FP16, "Unsupported low-precision VEC type");
        __tx_vec_cvt_fp16_fp32<0, 0, 0>();
    }
    __tx_vec_shuffle_compress<0, 0>(0);
    __tx_vec_lsu_v2u<0, 1>(scratch.address);
    start_vec_block(kInstructions);
    dayan_wait_vec();
    copy_packed_lowp<LsuGroup>(scratch, output, rows, chunks);
}

static inline void issue_sigmoid_mul_configured(const F32Tensor &           gate,
                                                const F32Tensor &           value,
                                                const F32Tensor &           output,
                                                const ActivationConstants & constants) {
    enum : long {
        kInstructions = 10,
    };

    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(gate.address);
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.neg_log2e);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 1>(value.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output.address);
    start_vec_block(kInstructions);
}

static inline void issue_sigmoid_mul(const F32Tensor &           gate,
                                     const F32Tensor &           value,
                                     const F32Tensor &           output,
                                     const ActivationConstants & constants,
                                     long                        rows,
                                     long                        columns) {
    configure_activation(gate, value, output, rows, columns);
    issue_sigmoid_mul_configured(gate, value, output, constants);
}

static inline void issue_silu_configured(const F32Tensor &           input,
                                         const F32Tensor &           output,
                                         const ActivationConstants & constants) {
    issue_sigmoid_mul_configured(input, input, output, constants);
}

static inline void issue_silu(const F32Tensor &           input,
                              const F32Tensor &           output,
                              const ActivationConstants & constants,
                              long                        rows,
                              long                        columns) {
    configure_activation(input, input, output, rows, columns);
    issue_silu_configured(input, output, constants);
}

template <long MatDataType = -1, bool PackedOutput = false>
static inline void issue_swiglu_configured(const F32Tensor &           gate,
                                           const F32Tensor &           up,
                                           const F32Tensor &           output) {
    enum : long {
        kInstructions = 11 + (MatDataType == -1 ? 0 : 2) + (PackedOutput ? 1 : 0),
    };

    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(gate.address);
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(-1.4426950408889634f));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_vv_fp32_mul<0, 0, 4, 0>();
    __tx_vec_lsu_u2v<1, 1>(up.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    if constexpr (MatDataType == BaseNPUInstBuilder::DT_BF16) {
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
    } else if constexpr (MatDataType == BaseNPUInstBuilder::DT_FP16) {
        __tx_vec_cvt_fp16_fp32<0, 0, 0>();
    }
    if constexpr (MatDataType != -1) {
        __tx_vec_shuffle_compress<0, 0>(0);
    }
    if constexpr (PackedOutput) {
        static_assert(MatDataType != -1, "Packed output requires low precision");
        __tx_vec_lsu_g2s<0>(0xffffffffU);
        __tx_vec_lsu_v2u_mask<0, 3, 0>(output.address);
    } else {
        __tx_vec_lsu_v2u<0, 3>(output.address);
    }
    start_vec_block(kInstructions);
}

template <long MatDataType = -1>
static inline void issue_swiglu(const F32Tensor &           gate,
                                const F32Tensor &           up,
                                const F32Tensor &           output,
                                long                        rows,
                                long                        columns) {
    configure_activation(gate, up, output, rows, columns);
    issue_swiglu_configured<MatDataType>(gate, up, output);
}

template <long MatDataType = -1, bool PackedOutput = false>
static inline void issue_geglu_configured(const F32Tensor &           gate,
                                          const F32Tensor &           up,
                                          const F32Tensor &           output) {
    enum : long {
        kInstructions = 27 + (MatDataType == -1 ? 0 : 2) + (PackedOutput ? 1 : 0),
    };

    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(gate.address);
    __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(0.044715f));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_vv_fp32_mul<0, 0, 3, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(0.7978845608028654f));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(-2.8853900817779268f));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(2.0f));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(1.0f));
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((long) f32_bits(0.5f));
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_vv_fp32_mul<0, 0, 13, 0>();
    __tx_vec_lsu_u2v<1, 1>(up.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    if constexpr (MatDataType == BaseNPUInstBuilder::DT_BF16) {
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
    } else if constexpr (MatDataType == BaseNPUInstBuilder::DT_FP16) {
        __tx_vec_cvt_fp16_fp32<0, 0, 0>();
    }
    if constexpr (MatDataType != -1) {
        __tx_vec_shuffle_compress<0, 0>(0);
    }
    if constexpr (PackedOutput) {
        static_assert(MatDataType != -1, "Packed output requires low precision");
        __tx_vec_lsu_g2s<0>(0xffffffffU);
        __tx_vec_lsu_v2u_mask<0, 3, 0>(output.address);
    } else {
        __tx_vec_lsu_v2u<0, 3>(output.address);
    }
    start_vec_block(kInstructions);
}

template <long MatDataType = -1>
static inline void issue_geglu(const F32Tensor &           gate,
                               const F32Tensor &           up,
                               const F32Tensor &           output,
                               long                        rows,
                               long                        columns) {
    configure_activation(gate, up, output, rows, columns);
    issue_geglu_configured<MatDataType>(gate, up, output);
}

}  // namespace ggml_dayan_vec_detail

#endif
