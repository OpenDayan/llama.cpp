#pragma once

#include "ggml-dayan-matmul.h"
#include "ggml-dayan-workers.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct SSMConvUramLayout {
    uint64_t input;
    uint64_t output;
    uint64_t staging;
    uint64_t total;
};

static inline SSMConvUramLayout ssm_conv_uram_layout(uint32_t tokens, uint32_t kernel_size, uint32_t channels) {
    const uint64_t weight_rows = align_up(kernel_size, 16);
    const uint64_t input_rows = align_up((uint64_t) tokens + kernel_size - 1U, 16);
    SSMConvUramLayout layout;
    layout.input   = weight_rows * channels * sizeof(float);
    layout.output  = layout.input + input_rows * channels * sizeof(float);
    layout.staging = layout.output + (uint64_t) tokens * channels * sizeof(float);
    layout.total   = layout.staging + input_rows * channels * sizeof(float);
    return layout;
}

static inline void load_ssm_transposed(uint64_t destination,
                                        uint64_t staging,
                                        const float * source,
                                        uint32_t columns,
                                        uint32_t channels,
                                        uint32_t channel_tile,
                                        uint64_t source_stride) {
    const uint64_t row_bytes = align_up(columns, 16) * sizeof(float);
    const uint64_t source_address = (uint64_t) source;
    const uint64_t output_stride = (uint64_t) channel_tile * sizeof(float);
    long checksum = 0;
    if (((source_address | source_stride) & 63U) == 0 && columns % 16U == 0 && channels % 16U == 0) {
        set_lsu_matrix<0, true, 2>(checksum, channels, (uint64_t) columns * sizeof(float), source_stride, output_stride);
        __tx_lsu_dma_d2u<0>(destination, source_address);
        return;
    }

    set_lsu_matrix<0>(checksum, channels, row_bytes, source_stride, row_bytes);
    load_matrix<0>((uint64_t) columns * sizeof(float), channels, row_bytes, channels, staging, source_address);
    dayan_wait_lsu();
    set_lsu_matrix<0, true, 2>(checksum, align_up(channels, 16), row_bytes, row_bytes, output_stride);
    __tx_lsu_dma_u2u<0>(destination, staging);
}

static inline void issue_ssm_silu_registers(const DayanSSMConvConfig * config) {
    __tx_vec_lsu_g2v_broadcast32<1, 2>(config->neg_log2e_bits);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>(config->one_bits);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_vv_fp32_mul<0, 0, 4, 0>();
}

static inline void issue_ssm_depthwise_f32(uint64_t input,
                                           uint64_t weight,
                                           uint64_t output,
                                           uint32_t tokens,
                                           uint32_t channels,
                                           uint32_t channel_tile,
                                           const DayanSSMConvConfig * config) {
    const uint32_t kernel_size = config->kernel_size;
    const bool silu = (config->flags & 1U) != 0 && kernel_size < kDayanSSMConvMaxKernelSize;
    const long chunks       = (channels + Constants::kVecLanes - 1U) / Constants::kVecLanes;
    const long row_stride   = (long) channel_tile * sizeof(float);
    const long instructions = 4L * kernel_size + (silu ? 7 : 0);

    ggml_dayan_vec_detail::configure_vec_grid(chunks, tokens);
    configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, row_stride);
    configure_vec_iterator<kVecV1Base>(0, 0, Constants::kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, row_stride);
    begin_vec_block(instructions);

    __tx_vec_lsu_u2v<0, 0>(input);
    __tx_vec_lsu_u2v<1, 1>(weight);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    for (uint32_t tap = 1; tap < kernel_size; ++tap) {
        __tx_vec_lsu_u2v<0, 0>(input + (uint64_t) tap * row_stride);
        __tx_vec_lsu_u2v<1, 1>(weight + (uint64_t) tap * row_stride);
        __tx_vec_vv_fp32_mul<1, 16, 0, 0>();
        __tx_vec_vv_fp32_add<0, 16, 1, 0>();
    }
    if (silu) {
        issue_ssm_silu_registers(config);
    }
    __tx_vec_lsu_v2u<0, 3>(output);
    start_vec_block(instructions);
}

static inline void store_ssm_output(float *  output,
                                    uint64_t source,
                                    uint32_t total_tokens,
                                    uint32_t token_count,
                                    uint32_t channels,
                                    uint32_t channel_base,
                                    uint32_t channel_count,
                                    uint32_t channel_tile,
                                    uint32_t sequence,
                                    uint32_t token_base) {
    const uint64_t target = reinterpret_cast<uint64_t>(
        output + ((uint64_t) sequence * total_tokens + token_base) * channels + channel_base);
    long checksum = 0;
    set_lsu_matrix<0>(checksum, token_count, (uint64_t) channel_tile * sizeof(float),
                      (uint64_t) channel_tile * sizeof(float), (uint64_t) channels * sizeof(float));
    store_matrix<0>((uint64_t) channel_count * sizeof(float), token_count,
                    (uint64_t) channel_tile * sizeof(float), token_count, source, target);
}

static inline void run_ssm_conv(const DayanSSMConvConfig * config,
                                const float *              input,
                                const float *              weight,
                                float *                    output,
                                const float *              qkv = nullptr,
                                float *                    state_output = nullptr, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {(uint64_t) config->channels * config->tokens * config->sequences, (uint64_t) ((config->channels + 63) / 64) * config->sequences * ((config->tokens + kDayanSSMConvTokenTile - 1) / kDayanSSMConvTokenTile), 256U * 1024U}, [&](const NativeContext & worker) {
            run_ssm_conv(config, input, weight, output, qkv, state_output, worker);
        })) { return; }

    DAYAN_PARAMS_IF(config == nullptr || input == nullptr || weight == nullptr || output == nullptr || config->input_length == 0 || config->channels == 0 || config->tokens == 0 ||
                    config->sequences == 0 || config->kernel_size == 0 ||
                    config->kernel_size > kDayanSSMConvMaxKernelSize ||
                    (uint64_t) config->input_length != (uint64_t) config->kernel_size - 1U + config->tokens ||
                    (config->flags & ~(kDayanSSMConvFlagSilu | kDayanSSMConvFlagNoStateCopy)) != 0 || !dayan_native_task_mode_valid() ||
                    context.tasks == 0U ||
                    context.task >= context.tasks ||
                    dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        DAYAN_PARAMS_ERROR();
    }
    DAYAN_PARAMS_IF(qkv != nullptr && (config->flags & kDayanSSMConvFlagNoStateCopy) == 0 &&
                    state_output == nullptr && config->kernel_size > 1) {
        DAYAN_PARAMS_ERROR();
    }
    static_assert(kDayanSSMConvTokenTile >= kDayanSSMConvMaxKernelSize, "SSM token tiles must cover the history");

    const uint64_t uram_address = native_uram(context);
    DAYAN_PARAMS_IF(uram_address == 0) {
        DAYAN_PARAMS_ERROR();
    }

    const bool update_state = qkv != nullptr && (config->flags & kDayanSSMConvFlagNoStateCopy) == 0;
    const uint32_t task = context.task;
    const uint32_t tasks = context.tasks;
    const uint32_t token_tile = config->tokens < kDayanSSMConvTokenTile ? config->tokens : kDayanSSMConvTokenTile;
    const uint64_t channel_groups = ((uint64_t) config->channels + Constants::kVecLanes - 1U) / Constants::kVecLanes;
    const uint32_t channel_tasks = channel_groups < tasks ? (uint32_t) channel_groups : tasks;
    const uint32_t channel_task = task % channel_tasks;
    const uint32_t row_task = task / channel_tasks;
    const uint32_t row_tasks = (tasks - 1U - channel_task) / channel_tasks + 1U;
    const uint64_t channel_begin = channel_groups * channel_task / channel_tasks * Constants::kVecLanes;
    const uint64_t channel_limit = channel_groups * (channel_task + 1U) / channel_tasks * Constants::kVecLanes;
    const uint64_t channel_end = channel_limit < config->channels ? channel_limit : config->channels;
    const uint64_t shard_channels = channel_end - channel_begin;
    const uint64_t bytes_per_channel = ssm_conv_uram_layout(token_tile, config->kernel_size, Constants::kVecLanes).total / Constants::kVecLanes;
    const uint64_t max_channel_tile = native_uram_size(context) / bytes_per_channel / Constants::kVecLanes * Constants::kVecLanes;
    DAYAN_PARAMS_IF(max_channel_tile == 0) {
        DAYAN_PARAMS_ERROR();
    }
    const uint64_t channel_tiles = (shard_channels + max_channel_tile - 1U) / max_channel_tile;
    const uint32_t channel_tile = (uint32_t) align_up((shard_channels + channel_tiles - 1U) / channel_tiles, Constants::kVecLanes);
    const uint64_t token_tiles = ((uint64_t) config->tokens + token_tile - 1U) / token_tile;

    for (uint64_t channel_base = channel_begin; channel_base < channel_end; channel_base += channel_tile) {
        const uint32_t channel_count = (uint32_t) (channel_end - channel_base < channel_tile ? channel_end - channel_base : channel_tile);
        const uint32_t tile_elements = (uint32_t) align_up(channel_count, Constants::kVecLanes);
        const SSMConvUramLayout layout = ssm_conv_uram_layout(token_tile, config->kernel_size, tile_elements);
        const uint64_t weight_uram = uram_address;
        const uint64_t input_uram = uram_address + layout.input;
        const uint64_t output_uram = uram_address + layout.output;
        const uint64_t staging_uram = uram_address + layout.staging;

        load_ssm_transposed(weight_uram, staging_uram, weight + channel_base * config->kernel_size,
                             config->kernel_size, channel_count, tile_elements, (uint64_t) config->kernel_size * sizeof(float));
        dayan_wait_lsu();

        for (uint64_t work = row_task; work < (uint64_t) config->sequences * token_tiles; work += row_tasks) {
            const uint32_t sequence = (uint32_t) (work / token_tiles);
            const uint32_t token_base = (uint32_t) ((work % token_tiles) * token_tile);
            const uint32_t token_count = config->tokens - token_base < token_tile ? config->tokens - token_base : token_tile;
            const uint32_t input_rows = token_count + config->kernel_size - 1U;
            const uint32_t history = config->kernel_size - 1U;
            const uint64_t row_stride = (uint64_t) tile_elements * sizeof(float);
            if (qkv != nullptr) {
                const uint32_t state_rows = token_base == 0 ? history : 0;
                if (state_rows != 0) {
                    load_ssm_transposed(input_uram, staging_uram,
                        input + ((uint64_t) sequence * config->channels + channel_base) * history,
                        history, channel_count, tile_elements, (uint64_t) history * sizeof(float));
                    dayan_wait_lsu();
                }
                const uint32_t qkv_token = token_base == 0 ? 0 : token_base - history;
                const uint32_t qkv_rows = input_rows - state_rows;
                const float * source = qkv + ((uint64_t) sequence * config->tokens + qkv_token) * config->channels + channel_base;
                long checksum = 0;
                set_lsu_matrix<0>(checksum, qkv_rows, row_stride, (uint64_t) config->channels * sizeof(float), row_stride);
                load_matrix<0>((uint64_t) channel_count * sizeof(float), qkv_rows, row_stride, qkv_rows,
                               input_uram + state_rows * row_stride, (uint64_t) source);
            } else {
                const float * source = input + ((uint64_t) sequence * config->channels + channel_base) * config->input_length + token_base;
                load_ssm_transposed(input_uram, staging_uram, source, input_rows, channel_count, tile_elements,
                                     (uint64_t) config->input_length * sizeof(float));
            }
            dayan_wait_lsu();

            issue_ssm_depthwise_f32(input_uram, weight_uram, output_uram, token_count, channel_count, tile_elements, config);
            if ((config->flags & 1U) != 0 && config->kernel_size == kDayanSSMConvMaxKernelSize) {
                // The 63-tap convolution leaves no room for SiLU in the 255-instruction block.
                dayan_wait_vec();
                begin_vec_block(9);
                __tx_vec_lsu_u2v<0, 0>(output_uram);
                issue_ssm_silu_registers(config);
                __tx_vec_lsu_v2u<0, 3>(output_uram);
                start_vec_block(9);
            }
            dayan_wait_vec();
            store_ssm_output(output, output_uram, config->tokens, token_count, config->channels, (uint32_t) channel_base,
                               channel_count, tile_elements, sequence, token_base);
            dayan_wait_lsu();
            if (update_state && token_base + token_count == config->tokens) {
                if (history == 0) {
                    continue;
                }
                const uint32_t old_rows = config->tokens < history ? history - config->tokens : 0;
                const uint32_t qkv_rows = history - old_rows;
                const uint32_t old_start = config->tokens < history ? config->tokens : 0;
                if (old_rows != 0) {
                    load_ssm_transposed(input_uram, staging_uram,
                        input + ((uint64_t) sequence * config->channels + channel_base) * history + old_start,
                        old_rows, channel_count, tile_elements, (uint64_t) history * sizeof(float));
                    dayan_wait_lsu();
                }
                if (qkv_rows != 0) {
                    const float * state_source = qkv + ((uint64_t) sequence * config->tokens + config->tokens - qkv_rows) * config->channels + channel_base;
                    long checksum = 0;
                    set_lsu_matrix<0>(checksum, qkv_rows, row_stride, (uint64_t) config->channels * sizeof(float), row_stride);
                    load_matrix<0>((uint64_t) channel_count * sizeof(float), qkv_rows, row_stride, qkv_rows,
                                   input_uram + old_rows * row_stride, (uint64_t) state_source);
                    dayan_wait_lsu();
                }
                const uint64_t state_stride = align_up(history, 16) * sizeof(float);
                long checksum = 0;
                set_lsu_matrix<0, true, 2>(checksum, align_up(history, 16), row_stride,
                                            row_stride, state_stride);
                __tx_lsu_dma_u2u<0>(staging_uram, input_uram);
                dayan_wait_lsu();
                set_lsu_matrix<0>(checksum, channel_count, state_stride, state_stride,
                                  history * sizeof(float));
                store_matrix<0>(history * sizeof(float), channel_count, state_stride, channel_count,
                                staging_uram, (uint64_t) (state_output +
                                ((uint64_t) sequence * config->channels + channel_base) * history));
                dayan_wait_lsu();
            }
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif
