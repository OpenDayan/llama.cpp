#pragma once

#if defined(__DAYAN_DEVICE__)

#    include <dayan_intrinsics.h>
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

static inline void configure_vec_dims(long rows, long chunks = 1) {
    configure_vec_grid(rows, chunks);
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
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

static inline void configure_activation(const F32Tensor & input0,
                                        const F32Tensor & input1,
                                        const F32Tensor & output,
                                        long              rows,
                                        long              columns) {
    const long chunks = (columns + kVecLanes - 1) / kVecLanes;

    configure_vec_grid(chunks, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, input0.row_stride);
    configure_vec_iterator<kVecV1Base>(0, 0, kVecBytes, input1.row_stride);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, output.row_stride);
}

template <long MatDataType>
static inline void issue_f32_to_lowp(const F32Tensor & input,
                                     const LowpTensor & output,
                                     long               rows,
                                     long               columns) {
    enum : long {
        kInstructions = 6,
        kLowpBytes    = kVecLanes * sizeof(uint16_t),
    };

    const long chunks = (columns + kVecLanes - 1) / kVecLanes;
    configure_vec_grid(chunks, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, input.row_stride);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kLowpBytes, output.row_stride);
    begin_vec_block(kInstructions);
    __tx_vec_s_bitwise<0, 3, 0, 3, 0>();
    __tx_vec_s_int32_add<0, 1, 0>();
    __tx_vec_lsu_u2v<0, 0>(input.address);
    if constexpr (MatDataType == BaseNPUInstBuilder::DT_BF16) {
        __tx_vec_cvt_bf16_fp32<0, 0, 0>();
    } else {
        static_assert(MatDataType == BaseNPUInstBuilder::DT_FP16, "Unsupported low-precision VEC type");
        __tx_vec_cvt_fp16_fp32<0, 0, 0>();
    }
    __tx_vec_shuffle_compress<0, 0>(1);
    __tx_vec_lsu_v2u_mask<3, 0, 0>(output.address);
    start_vec_block(kInstructions);
}

static inline void issue_sigmoid_mul(const F32Tensor &            gate,
                                     const F32Tensor &            value,
                                     const F32Tensor &            output,
                                     const ActivationConstants & constants,
                                     long                         rows,
                                     long                         columns) {
    enum : long {
        kInstructions = 10,
    };

    configure_activation(gate, value, output, rows, columns);
    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(gate.address);
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.neg_log2e);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 5, 0>();
    __tx_vec_lsu_u2v<1, 1>(value.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(output.address);
    start_vec_block(kInstructions);
}

static inline void issue_silu(const F32Tensor &            input,
                              const F32Tensor &            output,
                              const ActivationConstants & constants,
                              long                         rows,
                              long                         columns) {
    issue_sigmoid_mul(input, input, output, constants, rows, columns);
}

static inline void issue_swiglu(const F32Tensor &            gate,
                                const F32Tensor &            up,
                                const F32Tensor &            output,
                                const ActivationConstants & constants,
                                long                         rows,
                                long                         columns) {
    enum : long {
        kInstructions = 12,
    };

    configure_activation(gate, up, output, rows, columns);
    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(gate.address);
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.neg_log2e);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 5, 0>();
    __tx_vec_lsu_u2v<1, 0>(gate.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(up.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(output.address);
    start_vec_block(kInstructions);
}

static inline void issue_geglu(const F32Tensor &            gate,
                               const F32Tensor &            up,
                               const F32Tensor &            output,
                               const ActivationConstants & constants,
                               long                         rows,
                               long                         columns) {
    enum : long {
        kInstructions = 30,
    };

    configure_activation(gate, up, output, rows, columns);
    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(gate.address);
    __tx_vec_lsu_u2v<1, 0>(gate.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.gelu_a);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 0>(gate.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.sqrt_2_over_pi);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.neg_2_log2e);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 5, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.two);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_sub<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.half);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 0>(gate.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(up.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(output.address);
    start_vec_block(kInstructions);
}

}  // namespace ggml_dayan_vec_detail

#endif
