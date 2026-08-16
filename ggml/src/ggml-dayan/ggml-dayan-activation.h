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
    __tx_vec_tcsr_write<kVecSync>(0);
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

template <long Vb, long Vrs1, long Rm, long Operation>
static inline void emit_vec_convert() {
    constexpr uint32_t instruction = 0x8024000bU | ((uint32_t) Vb << 25) | ((uint32_t) Operation << 15) |
                                     ((uint32_t) Rm << 12) | ((uint32_t) Vrs1 << 7);

    // Clang 23 omits the convert subclass bits in tx.vec.cvt encodings.
    __asm__ volatile(".word %0" : : "i"(instruction));
}

template <long Vb, long Vrs1, long Rm>
static inline void vec_cvt_fp32_int32() {
    emit_vec_convert<Vb, Vrs1, Rm, 0>();
}

template <long Vb, long Vrs1, long Rm>
static inline void vec_cvt_fp32_fp16() {
    emit_vec_convert<Vb, Vrs1, Rm, 2>();
}

template <long Vb, long Vrs1, long Rm>
static inline void vec_cvt_fp32_bf16() {
    emit_vec_convert<Vb, Vrs1, Rm, 3>();
}

template <long Vb, long Vrs1, long Rm>
static inline void vec_cvt_fp16_fp32() {
    emit_vec_convert<Vb, Vrs1, Rm, 4>();
}

template <long Vb, long Vrs1, long Rm>
static inline void vec_cvt_bf16_fp32() {
    emit_vec_convert<Vb, Vrs1, Rm, 5>();
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

static inline void configure_f32_tile(uint64_t input0,
                                      uint64_t input1,
                                      uint64_t output,
                                      long     elements,
                                      long     input1_stride = kVecBytes) {
    configure_vec_grid(f32_chunks(elements));
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, input1_stride, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, 0);
    (void) input0;
    (void) input1;
    (void) output;
}

template <long Operation>
static inline void issue_f32_binary(uint64_t output, uint64_t left, uint64_t right, long elements) {
    constexpr long instructions = 4;

    configure_f32_tile(left, right, output, elements);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) left);
    __tx_vec_lsu_u2v<1, 1>((long) right);
    if constexpr (Operation == 0) {
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    } else if constexpr (Operation == 1) {
        __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    } else {
        static_assert(Operation == 2, "Unsupported FP32 VEC operation");
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_add(uint64_t output, uint64_t left, uint64_t right, long elements) {
    issue_f32_binary<0>(output, left, right, elements);
}

static inline void f32_sub(uint64_t output, uint64_t left, uint64_t right, long elements) {
    issue_f32_binary<1>(output, left, right, elements);
}

static inline void f32_mul(uint64_t output, uint64_t left, uint64_t right, long elements) {
    issue_f32_binary<2>(output, left, right, elements);
}

template <long Operation>
static inline void issue_f32_scalar_bits(uint64_t output, uint32_t scalar_bits, uint64_t input, long elements) {
    constexpr long instructions = 4;

    configure_f32_tile(input, 0, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) scalar_bits);
    if constexpr (Operation == 0) {
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    } else {
        static_assert(Operation == 2, "Unsupported scalar FP32 VEC operation");
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_scalar_add(uint64_t output, float scalar, uint64_t input, long elements) {
    issue_f32_scalar_bits<0>(output, f32_bits(scalar), input, elements);
}

static inline void f32_scalar_mul(uint64_t output, float scalar, uint64_t input, long elements) {
    issue_f32_scalar_bits<2>(output, f32_bits(scalar), input, elements);
}

static inline void f32_scalar_add_bits(uint64_t output, uint32_t scalar_bits, uint64_t input, long elements) {
    issue_f32_scalar_bits<0>(output, scalar_bits, input, elements);
}

static inline void f32_scalar_mul_bits(uint64_t output, uint32_t scalar_bits, uint64_t input, long elements) {
    issue_f32_scalar_bits<2>(output, scalar_bits, input, elements);
}

static inline void f32_scalar_sub_left(uint64_t output, float scalar, uint64_t input, long elements) {
    constexpr long instructions = 4;

    configure_f32_tile(input, 0, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) f32_bits(scalar));
    __tx_vec_lsu_u2v<1, 1>((long) input);
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

template <bool Maximum>
static inline void issue_f32_extreme(uint64_t output, uint64_t left, uint64_t right, long elements) {
    constexpr long instructions = 5;

    configure_f32_tile(left, right, output, elements);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) left);
    __tx_vec_lsu_u2v<1, 1>((long) right);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    if constexpr (Maximum) {
        __tx_vec_vv_mux2<0, 0, 16, 0>();
    } else {
        __tx_vec_vv_mux2<0, 16, 0, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_max(uint64_t output, uint64_t left, uint64_t right, long elements) {
    issue_f32_extreme<true>(output, left, right, elements);
}

static inline void f32_min(uint64_t output, uint64_t left, uint64_t right, long elements) {
    issue_f32_extreme<false>(output, left, right, elements);
}

template <bool Maximum>
static inline void issue_f32_scalar_extreme_bits(uint64_t output,
                                                 uint32_t scalar_bits,
                                                 uint64_t input,
                                                 long     elements) {
    constexpr long instructions = 5;

    configure_f32_tile(input, 0, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) scalar_bits);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    if constexpr (Maximum) {
        __tx_vec_vv_mux2<0, 0, 16, 0>();
    } else {
        __tx_vec_vv_mux2<0, 16, 0, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_scalar_max(uint64_t output, float scalar, uint64_t input, long elements) {
    issue_f32_scalar_extreme_bits<true>(output, f32_bits(scalar), input, elements);
}

static inline void f32_scalar_min(uint64_t output, float scalar, uint64_t input, long elements) {
    issue_f32_scalar_extreme_bits<false>(output, f32_bits(scalar), input, elements);
}

static inline void f32_scalar_max_bits(uint64_t output, uint32_t scalar_bits, uint64_t input, long elements) {
    issue_f32_scalar_extreme_bits<true>(output, scalar_bits, input, elements);
}

static inline void f32_scalar_min_bits(uint64_t output, uint32_t scalar_bits, uint64_t input, long elements) {
    issue_f32_scalar_extreme_bits<false>(output, scalar_bits, input, elements);
}

template <long Operation>
static inline void f32_sfu(uint64_t output, uint64_t input, long elements) {
    constexpr long instructions = 3;

    configure_f32_tile(input, 0, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, Operation>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

template <long Operation>
static inline void f32_scalar_sfu(uint64_t output, float input) {
    constexpr long instructions = 3;

    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) f32_bits(input));
    __tx_vec_v_fp32_sfu<0, 0, Operation>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_div(uint64_t output, uint64_t left, uint64_t right, long elements) {
    constexpr long instructions = 5;

    configure_f32_tile(left, right, output, elements);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) left);
    __tx_vec_lsu_u2v<1, 1>((long) right);
    __tx_vec_v_fp32_sfu<1, 16, 5>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_sqrt(uint64_t output, uint64_t input, long elements) {
    constexpr long instructions = 4;

    configure_f32_tile(input, 0, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 4>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

template <long Operation>
static inline void issue_f32_scalar_address(uint64_t output, uint64_t input, uint64_t scalar, long elements) {
    constexpr long instructions = 4;

    configure_f32_tile(input, scalar, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v<1, 1>((long) scalar);
    if constexpr (Operation == 0) {
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    } else if constexpr (Operation == 1) {
        __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    } else {
        static_assert(Operation == 2, "Unsupported scalar-address FP32 VEC operation");
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    }
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_add_scalar_address(uint64_t output, uint64_t input, uint64_t scalar, long elements) {
    issue_f32_scalar_address<0>(output, input, scalar, elements);
}

static inline void f32_sub_scalar_address(uint64_t output, uint64_t input, uint64_t scalar, long elements) {
    issue_f32_scalar_address<1>(output, input, scalar, elements);
}

static inline void f32_mul_scalar_address(uint64_t output, uint64_t input, uint64_t scalar, long elements) {
    issue_f32_scalar_address<2>(output, input, scalar, elements);
}

static inline void f32_fill_bits(uint64_t output, uint32_t value_bits, long elements) {
    constexpr long instructions = 2;

    configure_f32_tile(0, 0, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) value_bits);
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_fill_address(uint64_t output, uint64_t scalar, long elements) {
    constexpr long instructions = 2;

    configure_f32_tile(0, 0, output, elements, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) scalar);
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(instructions);
    dayan_wait_vec();
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
    dayan_cache_flush(reinterpret_cast<const void *>(mask_address), sizeof(mask));

    constexpr long instructions = 3;
    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
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
    dayan_cache_flush(reinterpret_cast<const void *>(mask_address), sizeof(mask));

    constexpr long instructions = 3;
    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2s<0, 0>((long) mask_address);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) scalar);
    __tx_vec_lsu_v2u_mask<0, 0, 0>((long) (output + (uint64_t) (elements / kVecLanes) * kVecBytes));
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void f32_int32_scalar(uint64_t output, uint32_t value) {
    constexpr long instructions = 3;

    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) value);
    vec_cvt_fp32_int32<0, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
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

template <bool Maximum>
static inline void f32_reduce(uint64_t output, uint64_t input, long elements) {
    const long chunks = f32_chunks(elements);
    constexpr long reduce_instructions = Maximum ? 20 : 14;

    configure_vec_grid(chunks);
    configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kVecBytes, 0);
    begin_vec_block(reduce_instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    if constexpr (Maximum) {
        emit_f32_max_stages();
    } else {
        emit_f32_sum_stages();
    }
    __tx_vec_lsu_v2u<0, 3>((long) output);
    start_vec_block(reduce_instructions);
    dayan_wait_vec();

    if (chunks == 1) {
        return;
    }

    const long combine_instructions = Maximum ? 3 * chunks - 1 : 2 * chunks;
    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(combine_instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) output);
    for (long chunk = 1; chunk < chunks; ++chunk) {
        __tx_vec_lsu_u2v_broadcast32<1, 1>((long) (output + (uint64_t) chunk * kVecBytes));
        if constexpr (Maximum) {
            __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
            __tx_vec_vv_mux2<0, 0, 16, 0>();
        } else {
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
        }
    }
    __tx_vec_lsu_v2u<0, 0>((long) output);
    start_vec_block(combine_instructions);
    dayan_wait_vec();
}

static inline void f32_reduce_sum(uint64_t output, uint64_t input, long elements) {
    f32_reduce<false>(output, input, elements);
}

static inline void f32_reduce_max(uint64_t output, uint64_t input, long elements) {
    f32_reduce<true>(output, input, elements);
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

template <long Group>
static inline void copy_packed_lowp(const F32Tensor & scratch,
                                    const LowpTensor & output,
                                    long               rows,
                                    long               chunks) {
    enum : long {
        kLowpBytes = kVecLanes * sizeof(uint16_t),
    };

    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(kLowpBytes / 64 - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(0);
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
    for (long row = 0; row < rows; ++row) {
        for (long chunk = 0; chunk < chunks; ++chunk) {
            __tx_lsu_dma_u2u<Group>(output.address + row * output.row_stride + chunk * kLowpBytes,
                                    scratch.address + row * scratch.row_stride + chunk * kVecBytes);
            dayan_wait_lsu();
        }
    }
}

template <long MatDataType, long LsuGroup>
static inline void convert_f32_to_lowp(const F32Tensor & input,
                                       const F32Tensor & scratch,
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
        vec_cvt_bf16_fp32<0, 0, 0>();
    } else {
        static_assert(MatDataType == BaseNPUInstBuilder::DT_FP16, "Unsupported low-precision VEC type");
        vec_cvt_fp16_fp32<0, 0, 0>();
    }
    __tx_vec_shuffle_compress<0, 0>(1);
    __tx_vec_lsu_v2u<0, 1>(scratch.address);
    start_vec_block(kInstructions);
    dayan_wait_vec();
    copy_packed_lowp<LsuGroup>(scratch, output, rows, chunks);
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
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 1>(value.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output.address);
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
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 0>(gate.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(up.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output.address);
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
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.two);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.one);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants.half);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 0>(gate.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(up.address);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(output.address);
    start_vec_block(kInstructions);
}

}  // namespace ggml_dayan_vec_detail

#endif
