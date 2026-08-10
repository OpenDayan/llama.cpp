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
    uint64_t input;
    uint64_t weight;
    uint64_t output;
    uint64_t constants;
    uint64_t total;
};

static inline uint16_t ssm_conv_bf16(float value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.f = value;
    bits.u += 0x7fffU + ((bits.u >> 16) & 1U);
    return (uint16_t) (bits.u >> 16);
}

static inline SSMConvUramLayout ssm_conv_uram_layout(uint32_t tokens, uint32_t kernel_size, uint32_t channels) {
    SSMConvUramLayout layout;
    const uint64_t    input_rows   = (uint64_t) tokens + kernel_size - 1U;
    const uint64_t    input_bytes  = input_rows * channels * sizeof(uint16_t);
    const uint64_t    weight_bytes = (uint64_t) kernel_size * channels * channels * sizeof(uint16_t);
    const uint64_t    output_bytes = (uint64_t) tokens * channels * sizeof(float);

    layout.input     = 0;
    layout.weight    = align_up(layout.input + input_bytes, Constants::kUramAlignment);
    layout.output    = align_up(layout.weight + weight_bytes, Constants::kUramAlignment);
    layout.constants = align_up(layout.output + output_bytes, Constants::kUramAlignment);
    layout.total     = align_up(layout.constants + Constants::kConstantStride, Constants::kUramAlignment);
    return layout;
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

static inline void issue_silu_f32(uint64_t input,
                                  uint64_t output,
                                  uint64_t constants,
                                  uint32_t rows,
                                  uint32_t columns) {
    enum : long {
        kInstructions = 10,
    };

    const long row_stride = (long) columns * sizeof(float);
    const long chunks     = (columns + Constants::kVecLanes - 1U) / Constants::kVecLanes;
    configure_vec_dims(rows, chunks);
    configure_vec_iterator<kVecV0Base>(0, 0, row_stride, Constants::kVecBytes);
    configure_vec_iterator<kVecV1Base>(0, 0, row_stride, Constants::kVecBytes);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants + sizeof(float));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 5, 0>();
    __tx_vec_lsu_u2v<1, 0>(input);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(output);
    start_vec_block(kInstructions);
}

static inline void pack_ssm_conv_weight(uint16_t *    destination,
                                        const float * weight,
                                        uint32_t      kernel_size,
                                        uint32_t      channel_base,
                                        uint32_t      channel_count,
                                        uint32_t      channel_tile) {
    const uint64_t elements = (uint64_t) kernel_size * channel_tile * channel_tile;
    for (uint64_t index = 0; index < elements; ++index) {
        destination[index] = 0;
    }
    for (uint32_t tap = 0; tap < kernel_size; ++tap) {
        const uint64_t tap_base = (uint64_t) tap * channel_tile * channel_tile;
        for (uint32_t channel = 0; channel < channel_count; ++channel) {
            const uint64_t source = (uint64_t) (channel_base + channel) * kernel_size + tap;
            destination[tap_base + (uint64_t) channel * channel_tile + channel] = ssm_conv_bf16(weight[source]);
        }
    }
}

static inline void pack_ssm_conv_input(uint16_t *    destination,
                                       const float * input,
                                       uint32_t      input_length,
                                       uint32_t      channels,
                                       uint32_t      sequence,
                                       uint32_t      token_base,
                                       uint32_t      rows,
                                       uint32_t      channel_base,
                                       uint32_t      channel_count,
                                       uint32_t      channel_tile) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t channel = 0; channel < channel_tile; ++channel) {
            uint16_t value = 0;
            if (channel < channel_count) {
                const uint64_t source =
                    ((uint64_t) sequence * channels + channel_base + channel) * input_length + token_base + row;
                value = ssm_conv_bf16(input[source]);
            }
            destination[(uint64_t) row * channel_tile + channel] = value;
        }
    }
}

static inline void run_ssm_conv(const DayanSSMConvConfig * config,
                                const float *              input,
                                const float *              weight,
                                float *                    output,
                                DayanSSMConvPerf *         perf) {
    if (config == nullptr || input == nullptr || weight == nullptr || output == nullptr || perf == nullptr ||
        config->input_length == 0 || config->channels == 0 || config->tokens == 0 || config->sequences == 0 ||
        config->kernel_size == 0 ||
        (uint64_t) config->input_length != (uint64_t) config->kernel_size - 1U + config->tokens ||
        (config->flags & ~1U) != 0 || dayan_task_mode() != 1U || dayan_task_dim(DAYAN_TASK_X) != 1U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        dayan_proc_exit(-22);
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
        uint16_t *              input_uram    = (uint16_t *) ((char *) uram + layout.input);
        uint16_t *              weight_uram   = (uint16_t *) ((char *) uram + layout.weight);
        float *                 output_uram   = (float *) ((char *) uram + layout.output);
        float *                 constants     = (float *) ((char *) uram + layout.constants);
        constants[0]                          = -1.4426950408889634f;
        constants[1]                          = 1.0f;
        pack_ssm_conv_weight(weight_uram, weight, config->kernel_size, channel_base, channel_count, channel_tile);

        for (uint32_t sequence = 0; sequence < config->sequences; ++sequence) {
            for (uint32_t token_base = 0; token_base < config->tokens; token_base += token_tile) {
                const uint32_t token_count =
                    config->tokens - token_base < token_tile ? config->tokens - token_base : token_tile;
                const uint32_t input_rows = token_count + config->kernel_size - 1U;
                pack_ssm_conv_input(input_uram, input, config->input_length, config->channels, sequence, token_base,
                                    input_rows, channel_base, channel_count, channel_tile);
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
                    const uint64_t output_address = uram_address + layout.output;
                    issue_silu_f32(output_address, output_address, uram_address + layout.constants, token_count,
                                   channel_tile);
                    dayan_wait_vec();
                }
                for (uint32_t token = 0; token < token_count; ++token) {
                    const uint64_t target =
                        ((uint64_t) sequence * config->tokens + token_base + token) * config->channels + channel_base;
                    const uint64_t source = (uint64_t) token * channel_tile;
                    for (uint32_t channel = 0; channel < channel_count; ++channel) {
                        output[target + channel] = output_uram[source + channel];
                    }
                }
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
