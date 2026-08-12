#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

#    include <npu_cache.h>

namespace ggml_dayan_matmul_detail {

enum : uint32_t {
    kSSMConvTokenTile        = 64,
    kSSMConvChannelTileSmall = 128,
    kSSMConvChannelTileLarge = 256,
};

struct SSMConvUramLayout {
    uint64_t input_f32;
    uint64_t input;
    uint64_t weight;
    uint64_t output;
    uint64_t zero;
    uint64_t constants;
    uint64_t total;
};

static inline SSMConvUramLayout ssm_conv_uram_layout(uint32_t tokens, uint32_t kernel_size, uint32_t channels) {
    SSMConvUramLayout layout;
    const uint64_t input_rows     = (uint64_t) tokens + kernel_size - 1U;
    const uint64_t input_f32_bytes = input_rows * channels * sizeof(float);
    const uint64_t input_bytes    = input_rows * channels * sizeof(uint16_t);
    const uint64_t weight_bytes   = (uint64_t) kernel_size * channels * channels * sizeof(uint16_t);
    const uint64_t output_bytes   = (uint64_t) tokens * channels * sizeof(float);

    layout.input_f32 = 0;
    layout.input     = align_up(layout.input_f32 + input_f32_bytes, Constants::kUramAlignment);
    layout.weight    = align_up(layout.input + input_bytes, Constants::kUramAlignment);
    layout.output    = align_up(layout.weight + weight_bytes, Constants::kUramAlignment);
    layout.zero      = align_up(layout.output + output_bytes, 64);
    layout.constants = layout.zero + 64;
    layout.total     = align_up(layout.constants + Constants::kConstantStride, Constants::kUramAlignment);
    return layout;
}

template <long Group>
static inline void configure_ssm_lsu(uint32_t dim0,
                                     uint32_t dim1,
                                     uint64_t input_dim0_stride,
                                     uint64_t input_dim1_stride,
                                     uint64_t output_dim0_stride,
                                     uint64_t output_dim1_stride,
                                     uint32_t read_bytes,
                                     uint32_t write_bytes) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(dim0 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(dim1 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(input_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(output_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
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
static inline void configure_ssm_convert(uint32_t rows,
                                         uint32_t columns,
                                         uint64_t input_row_stride,
                                         uint64_t output_row_stride) {
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM0>(columns - 1U);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM1>(rows - 1U);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(input_row_stride);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(output_row_stride);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(0);
}

static inline void issue_ssm_zero(uint64_t output, uint64_t zero, uint64_t bytes) {
    configure_ssm_lsu<0>(bytes / 64U, 1, 0, 0, 64, 0, 64, 64);
    __tx_lsu_dma_u2u<0>(output, zero);
}

template <long Group> static inline void configure_ssm_conv_bf16(uint32_t tokens, uint32_t channels) {
    const uint64_t input_row_stride  = (uint64_t) channels * sizeof(uint16_t);
    const uint64_t output_row_stride = (uint64_t) channels * sizeof(float);

    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKCI>(channels);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKWI>(tokens);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKHI>(1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKWO>(tokens);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKHO>(1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKWK>(1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKHK>(1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_TASKCO>(channels);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_WISTRIDE>(input_row_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_HISTRIDE>((uint64_t) tokens * input_row_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_WKSTRIDE>(input_row_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_HKSTRIDE>(input_row_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_COSTRIDE>(input_row_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_WOSTRIDE>(output_row_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_HOSTRIDE>((uint64_t) tokens * output_row_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ISWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_KSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_OSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DATATYPE>(Bf16Format::kMatDataType);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_PADL>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_PADU>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DILATIONW>(1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DILATIONH>(1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_STRIDEW>(1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_STRIDEH>(1);
}

static inline void issue_silu_f32(uint64_t input, uint64_t output, uint64_t constants, uint32_t rows, uint32_t columns) {
    const long row_stride = (long) columns * sizeof(float);
    ggml_dayan_vec_detail::issue_silu({ input, row_stride }, { output, row_stride },
                                      { constants, constants + sizeof(float), 0, 0, 0, 0, 0 }, rows, columns);
}

static inline void pack_ssm_conv_weight(uint64_t      input_f32,
                                        uint64_t      input_bf16,
                                        uint64_t      destination,
                                        const float * weight,
                                        uint32_t      kernel_size,
                                        uint32_t      channel_base,
                                        uint32_t      channel_count,
                                        uint32_t      channel_tile) {
    for (uint32_t tap = 0; tap < kernel_size; ++tap) {
        configure_ssm_lsu<0>(channel_count, 1, (uint64_t) kernel_size * sizeof(float), 0, sizeof(float), 0,
                             sizeof(float), sizeof(float));
        const float * source = weight + (uint64_t) channel_base * kernel_size + tap;
        __tx_lsu_dma_d2u<0>(input_f32, reinterpret_cast<uint64_t>(source));
        dayan_wait_lsu();

        configure_ssm_convert<0>(1, channel_count, 0, 0);
        __tx_macro_convert_fp32_bf16<0>((void *) input_bf16, (void *) input_f32);
        dayan_wait_macro();

        configure_ssm_lsu<0>(channel_count, 1, sizeof(uint16_t), 0,
                             (uint64_t) (channel_tile + 1U) * sizeof(uint16_t), 0, sizeof(uint16_t), sizeof(uint16_t));
        const uint64_t tap_base = destination + (uint64_t) tap * channel_tile * channel_tile * sizeof(uint16_t);
        __tx_lsu_dma_u2u<0>(tap_base, input_bf16);
        dayan_wait_lsu();
    }
}

static inline void pack_ssm_conv_input(uint64_t      input_f32,
                                       uint64_t      input_bf16,
                                       const float * input,
                                       uint32_t      input_length,
                                       uint32_t      channels,
                                       uint32_t      sequence,
                                       uint32_t      token_base,
                                       uint32_t      rows,
                                       uint32_t      channel_base,
                                       uint32_t      channel_count,
                                       uint32_t      channel_tile) {
    const float * source = input + ((uint64_t) sequence * channels + channel_base) * input_length + token_base;
    configure_ssm_lsu<0>(channel_count, rows, (uint64_t) input_length * sizeof(float), sizeof(float), sizeof(float),
                         (uint64_t) channel_tile * sizeof(float), sizeof(float), sizeof(float));
    __tx_lsu_dma_d2u<0>(input_f32, reinterpret_cast<uint64_t>(source));
    dayan_wait_lsu();

    configure_ssm_convert<0>(rows, channel_count, (uint64_t) channel_tile * sizeof(float),
                             (uint64_t) channel_tile * sizeof(uint16_t));
    __tx_macro_convert_fp32_bf16<0>((void *) input_bf16, (void *) input_f32);
    dayan_wait_macro();
}

static inline void run_ssm_conv(const DayanSSMConvConfig * config,
                                const float *              input,
                                const float *              weight,
                                float *                    output,
                                DayanSSMConvPerf *         perf) {
    DAYAN_PARAMS_IF(config == nullptr || input == nullptr || weight == nullptr || output == nullptr || perf == nullptr ||
                    config->input_length == 0 || config->channels == 0 || config->tokens == 0 ||
                    config->sequences == 0 || config->kernel_size == 0 ||
                    (uint64_t) config->input_length != (uint64_t) config->kernel_size - 1U + config->tokens ||
                    (config->flags & ~1U) != 0 || dayan_task_mode() != 1U ||
                    dayan_task_dim(DAYAN_TASK_X) != 1U || dayan_task_dim(DAYAN_TASK_Y) != 1U ||
                    dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        DAYAN_PARAMS_ERROR();
    }

    void * uram = dayan_uram_base();
    if (uram == nullptr) {
        dayan_proc_exit(-12);
    }

    const uint64_t          start_cycle  = read_cycle();
    const uint32_t          token_tile   = config->tokens < kSSMConvTokenTile ? config->tokens : kSSMConvTokenTile;
    const uint64_t          uram_size    = dayan_uram_size();
    const uint64_t          uram_address = (uint64_t) (uintptr_t) uram;
    const uint64_t          uram_offset  = uram_address & (Constants::kUramWindowStride - 1U);
    const SSMConvUramLayout small_layout =
        ssm_conv_uram_layout(token_tile, config->kernel_size, kSSMConvChannelTileSmall);
    if (small_layout.total > uram_size) {
        dayan_proc_exit(-12);
    }
    const SSMConvUramLayout large_layout =
        ssm_conv_uram_layout(token_tile, config->kernel_size, kSSMConvChannelTileLarge);
    const uint32_t primary_channel_tile =
        config->channels > kSSMConvChannelTileSmall && large_layout.total <= uram_size ? kSSMConvChannelTileLarge :
                                                                                         kSSMConvChannelTileSmall;

    cache_inval_range(input, (size_t) config->input_length * config->channels * config->sequences * sizeof(float));
    cache_inval_range(weight, (size_t) config->kernel_size * config->channels * sizeof(float));

    for (uint32_t channel_base = 0; channel_base < config->channels;) {
        const uint32_t remaining = config->channels - channel_base;
        const uint32_t channel_tile =
            primary_channel_tile == kSSMConvChannelTileLarge && remaining > kSSMConvChannelTileSmall ?
                kSSMConvChannelTileLarge :
                kSSMConvChannelTileSmall;
        const uint32_t          channel_count = remaining < channel_tile ? remaining : channel_tile;
        const SSMConvUramLayout layout        = ssm_conv_uram_layout(token_tile, config->kernel_size, channel_tile);
        const uint64_t input_f32_uram = uram_address + layout.input_f32;
        const uint64_t input_uram     = uram_address + layout.input;
        const uint64_t weight_uram    = uram_address + layout.weight;
        const uint64_t output_uram    = uram_address + layout.output;
        const uint64_t zero           = uram_address + layout.zero;
        const uint64_t constants      = uram_address + layout.constants;

        configure_ssm_lsu<0>(1, 1, 0, 0, 0, 0, sizeof(uint32_t), 64);
        __tx_lsu_dma_d2u<0>(zero, reinterpret_cast<uint64_t>(&config->zero_bits));
        configure_ssm_lsu<0>(1, 1, 0, 0, 0, 0, 2U * sizeof(uint32_t), 2U * sizeof(uint32_t));
        __tx_lsu_dma_d2u<0>(constants, reinterpret_cast<uint64_t>(&config->neg_log2e_bits));
        dayan_wait_lsu();
        issue_ssm_zero(weight_uram, zero,
                       (uint64_t) config->kernel_size * channel_tile * channel_tile * sizeof(uint16_t));
        dayan_wait_lsu();
        pack_ssm_conv_weight(input_f32_uram, input_uram, weight_uram, weight, config->kernel_size, channel_base,
                             channel_count, channel_tile);

        for (uint32_t sequence = 0; sequence < config->sequences; ++sequence) {
            for (uint32_t token_base = 0; token_base < config->tokens; token_base += token_tile) {
                const uint32_t token_count =
                    config->tokens - token_base < token_tile ? config->tokens - token_base : token_tile;
                const uint32_t input_rows = token_count + config->kernel_size - 1U;
                issue_ssm_zero(input_uram, zero, (uint64_t) input_rows * channel_tile * sizeof(uint16_t));
                dayan_wait_lsu();
                pack_ssm_conv_input(input_f32_uram, input_uram, input, config->input_length, config->channels,
                                    sequence, token_base, input_rows, channel_base, channel_count, channel_tile);
                configure_ssm_conv_bf16<0>(token_count, channel_tile);

                const uint64_t output_mat       = uram_offset + layout.output;
                const uint64_t input_row_bytes  = (uint64_t) channel_tile * sizeof(uint16_t);
                const uint64_t weight_tap_bytes = (uint64_t) channel_tile * channel_tile * sizeof(uint16_t);
                __tx_mat_conv2d<0>((long) output_mat, (long) (uram_offset + layout.input),
                                   (long) (uram_offset + layout.weight));
                for (uint32_t tap = 1; tap < config->kernel_size; ++tap) {
                    __tx_mat_conv2da<0>((long) output_mat,
                                        (long) (uram_offset + layout.input + (uint64_t) tap * input_row_bytes),
                                        (long) (uram_offset + layout.weight + (uint64_t) tap * weight_tap_bytes));
                }
                dayan_wait_mat();

                if ((config->flags & 1U) != 0) {
                    issue_silu_f32(output_uram, output_uram, constants, token_count, channel_tile);
                    dayan_wait_vec();
                }
                const uint64_t target = reinterpret_cast<uint64_t>(
                    output + ((uint64_t) sequence * config->tokens + token_base) * config->channels + channel_base);
                long checksum = 0;
                set_lsu_matrix<0>(checksum, token_count, (uint64_t) channel_tile * sizeof(float),
                                  (uint64_t) channel_tile * sizeof(float), (uint64_t) config->channels * sizeof(float));
                store_matrix<0>((uint64_t) channel_count * sizeof(float), token_count,
                                (uint64_t) channel_tile * sizeof(float), token_count, output_uram, target);
                dayan_wait_lsu();
            }
        }
        channel_base += channel_count;
    }
    cache_bypass_flush();
    perf->magic  = kDayanSSMConvPerfMagic;
    perf->cycles = read_cycle() - start_cycle;
    cache_bypass_flush();
}

}  // namespace ggml_dayan_matmul_detail

#endif
