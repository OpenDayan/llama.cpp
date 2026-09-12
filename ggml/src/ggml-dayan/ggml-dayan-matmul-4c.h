#pragma once

#include "ggml-dayan-matmul.h"
#include "ggml-dayan-ffn.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_4c_detail {

struct Constants {
    enum : int {
        kClusterCores = 4,
        kUramWindowStride = 2 * 1024 * 1024,
    };
};

using ggml_dayan_matmul_detail::min_int;
using ClusterShared = ggml_dayan_matmul_detail::MatmulCluster;

static inline ClusterShared & cluster_shared() {
    return *(ClusterShared *) shared;
}

static inline void initialize_resources() {
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    if (!shared || lane >= 4) {
        dayan_thread_exit(-22);
    }
    cluster_shared().phases[lane] = {};
    if (lane == 0) {
        for (int i = 0; i < 12; ++i) {
            cluster_shared().shared_phase[i] = 0;
        }
    }
    ggml_dayan_matmul_detail::matmul_check(dayan_cluster_barrier(4, &cluster_shared().barrier));
}

static inline int finish_epilogue_4c()
{
    return dayan_cluster_barrier(Constants::kClusterCores,
                                 &cluster_shared().barrier);
}

static inline bool moe_allocation_fits_4c(uint64_t element_size,
                                          uint64_t ne0, uint64_t ne1,
                                          uint64_t ne2)
{
    const uint64_t dimensions[3] = {ne0, ne1, ne2};
    uint64_t bytes = element_size;

    for (int index = 0; index < 3; ++index) {
        if (!dimensions[index] || dimensions[index] > UINT64_MAX / bytes)
            return false;
        bytes *= dimensions[index];
    }
    return true;
}

static inline bool mul_mat_id_config_valid_4c(
    const DayanMulMatIdConfig *config, uint64_t output_element_size)
{
    return config && config->k && config->n && config->experts &&
        config->experts <= INT32_MAX && config->experts_used &&
        config->tokens && config->activation_slots &&
        config->ids_row_stride >= (uint64_t) config->experts_used * sizeof(int32_t) &&
        config->ids_row_stride <= UINT64_MAX / config->tokens &&
        config->experts_used % config->activation_slots == 0U &&
        config->n % 16U == 0U &&
        moe_allocation_fits_4c(sizeof(uint16_t), config->k, config->n,
                               config->experts) &&
        moe_allocation_fits_4c(sizeof(uint16_t), config->k,
                               config->activation_slots, config->tokens) &&
        moe_allocation_fits_4c(sizeof(int32_t), config->experts_used,
                               config->tokens, 1U) &&
        moe_allocation_fits_4c(output_element_size, config->n,
                               config->experts_used, config->tokens);
}

static inline bool same_mul_mat_id_selection_4c(
    const DayanMulMatIdConfig &first, const DayanMulMatIdConfig &second)
{
    return first.k == second.k && first.experts == second.experts &&
        first.experts_used == second.experts_used &&
        first.tokens == second.tokens &&
        first.ids_row_stride == second.ids_row_stride &&
        first.activation_slots == second.activation_slots;
}

static int read_selected_expert_4c(const DayanMulMatIdConfig *config,
                                   const int32_t *ids, uint64_t index,
                                   int32_t *expert)
{
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    ClusterShared &cluster = cluster_shared();

    if (lane == 0U) {
        cluster.selected_status = 0;
        const uint64_t scratch = (uint64_t)(uintptr_t)dayan_uram_base();
        if (!scratch || dayan_uram_size() < sizeof(int32_t)) {
            cluster.selected_status = -12;
        } else {
            long checksum = 0;
            set_lsu_matrix<0>(checksum, 1, 64, 64, 64);
            ggml_dayan_matmul_detail::load_matrix<0>(
                sizeof(int32_t), 1, 64, 1, scratch,
                (uint64_t)(uintptr_t)ids + (index / config->experts_used) * config->ids_row_stride +
                    (index % config->experts_used) * sizeof(int32_t));
            dayan_wait_lsu();
            cluster.selected_expert = *(const int32_t *)(uintptr_t)scratch;
            if (cluster.selected_expert < 0 ||
                (uint32_t)cluster.selected_expert >= config->experts)
                cluster.selected_status = -22;
        }
    }
    const int status = dayan_cluster_barrier(Constants::kClusterCores,
                                             &cluster.barrier);
    if (status < 0)
        return status;
    *expert = cluster.selected_expert;
    return cluster.selected_status;
}

static inline DayanMatmulConfig moe_matmul_config_4c(
    const DayanMulMatIdConfig *config, uint64_t output_element_size)
{
    return {
        1U,
        config->n,
        config->k,
        (uint64_t)config->k * sizeof(uint16_t),
        (uint64_t)config->k * sizeof(uint16_t),
        (uint64_t)config->n * output_element_size,
    };
}

static inline uint64_t moe_activation_offset_4c(
    const DayanMulMatIdConfig *config, uint32_t token, uint32_t used)
{
    const uint32_t slot = used % config->activation_slots;
    return ((uint64_t)token * config->activation_slots + slot) * config->k;
}

static inline uint64_t moe_output_offset_4c(
    const DayanMulMatIdConfig *config, uint32_t token, uint32_t used)
{
    return ((uint64_t)token * config->experts_used + used) * config->n;
}

template <typename Format, bool Inference = false>
static int run_matmul_4c(const DayanMatmulConfig * config, const uint16_t * a, const uint16_t * bt, float * output) {
    return ggml_dayan_matmul_detail::run_matmul<Format, 4, 1, sizeof(float), false, ggml_dayan_matmul_detail::epilogue_identity, nullptr, 0, Inference>(config, a, bt, nullptr, output);
}

template <typename Format, bool Inference = false>
static int run_matmul_silu_4c(const DayanMatmulConfig * config, const uint16_t * a, const uint16_t * bt, float * output) {
    return ggml_dayan_matmul_detail::run_matmul<Format, 4, 1, sizeof(float), true, ggml_dayan_matmul_detail::epilogue_silu, nullptr, 0, Inference>(config, a, bt, nullptr, output);
}

struct PostprocessLayout {
    enum : uint64_t {
        kRows       = 128,
        kColumns    = 384,
        kF32Bytes   = kRows * kColumns * sizeof(float),
        kLowpBytes  = kRows * kColumns * sizeof(uint16_t),
        kInput0     = 0,
        kInput1     = kInput0 + kF32Bytes,
        kOutput     = kInput1 + kF32Bytes,
        kConstants  = kOutput + kLowpBytes,
        kTotalBytes = kConstants +
                      ggml_dayan_matmul_detail::Constants::kConstantCount *
                          ggml_dayan_matmul_detail::Constants::kConstantStride,
    };
};

static_assert(PostprocessLayout::kTotalBytes <= MATMUL_DAYAN_4C_URAM_BYTES,
              "4-Core matmul postprocess layout is too large");

static inline uint64_t postprocess_uram(uint64_t offset)
{
    return (uint64_t)(uintptr_t)dayan_uram_base() + offset;
}

static inline void initialize_postprocess_constants()
{
    const uint64_t base = postprocess_uram(PostprocessLayout::kConstants);
    const uint64_t stride = ggml_dayan_matmul_detail::Constants::kConstantStride;
    *(volatile float *)(uintptr_t)(base + 0U * stride) = -1.4426950408889634f;
    *(volatile float *)(uintptr_t)(base + 1U * stride) = 1.0f;
    *(volatile float *)(uintptr_t)(base + 2U * stride) = 0.044715f;
    *(volatile float *)(uintptr_t)(base + 3U * stride) = 0.7978845608028654f;
    *(volatile float *)(uintptr_t)(base + 4U * stride) = -2.8853900817779268f;
    *(volatile float *)(uintptr_t)(base + 5U * stride) = 2.0f;
    *(volatile float *)(uintptr_t)(base + 6U * stride) = 0.5f;
}

static inline ggml_dayan_vec_detail::ActivationConstants postprocess_constants()
{
    const uint64_t base = postprocess_uram(PostprocessLayout::kConstants);
    const uint64_t stride = ggml_dayan_matmul_detail::Constants::kConstantStride;
    return {
        base + 0U * stride,
        base + 1U * stride,
        base + 2U * stride,
        base + 3U * stride,
        base + 4U * stride,
        base + 5U * stride,
        base + 6U * stride,
    };
}

static inline void configure_postprocess_load(long group, uint64_t input_stride)
{
    long checksum = 0;
    if (group == 0) {
        set_lsu_matrix<0>(checksum, PostprocessLayout::kRows,
                          PostprocessLayout::kColumns * sizeof(float),
                          (long)input_stride,
                          PostprocessLayout::kColumns * sizeof(float));
    } else {
        set_lsu_matrix<1>(checksum, PostprocessLayout::kRows,
                          PostprocessLayout::kColumns * sizeof(float),
                          (long)input_stride,
                          PostprocessLayout::kColumns * sizeof(float));
    }
}

static inline void load_postprocess_tile(long group, uint64_t destination,
                                         const float *source, int rows,
                                         int columns)
{
    if (group == 0) {
        ggml_dayan_matmul_detail::load_matrix<0>(
            (uint64_t)columns * sizeof(float), rows,
            PostprocessLayout::kColumns * sizeof(float),
            PostprocessLayout::kRows, destination,
            (uint64_t)(uintptr_t)source);
    } else {
        ggml_dayan_matmul_detail::load_matrix<1>(
            (uint64_t)columns * sizeof(float), rows,
            PostprocessLayout::kColumns * sizeof(float),
            PostprocessLayout::kRows, destination,
            (uint64_t)(uintptr_t)source);
    }
}

static inline void configure_postprocess_store(uint64_t output_stride)
{
    long checksum = 0;
    set_lsu_matrix<3>(checksum, PostprocessLayout::kRows,
                      PostprocessLayout::kColumns * sizeof(uint16_t),
                      PostprocessLayout::kColumns * sizeof(uint16_t),
                      (long)output_stride);
}

static inline void store_postprocess_tile(uint16_t *destination, int rows,
                                          int columns)
{
    ggml_dayan_matmul_detail::store_matrix<3>(
        (uint64_t)columns * sizeof(uint16_t), rows,
        PostprocessLayout::kColumns * sizeof(uint16_t),
        PostprocessLayout::kRows,
        postprocess_uram(PostprocessLayout::kOutput),
        (uint64_t)(uintptr_t)destination);
    dayan_wait_lsu();
}

template <typename Format, bool Inference = false>
static int run_matmul_quantized_4c(const DayanMatmulConfig * config, const uint16_t * a, const uint16_t * bt, uint16_t * output) {
    return ggml_dayan_matmul_detail::run_matmul<Format, 4, 1, sizeof(uint16_t), true, ggml_dayan_matmul_detail::epilogue_lowp<Format>, nullptr, 0, Inference>(config, a, bt, nullptr, output);
}

template <typename Format, bool GeGLU, bool Inference = false>
static int run_matmul_glu_4c(const DayanMatmulConfig * config, const uint16_t * a, const uint16_t * gate, const uint16_t * up, uint16_t * output) {
    return ggml_dayan_ffn_detail::run<Format, 4, GeGLU, Inference>(config, a, gate, up, output);
}

template <typename Format>
static int run_matmul_pair_4c(const DayanMatmulPairConfig *config,
                              const uint16_t *a, const uint16_t *first_bt,
                              const uint16_t *second_bt, float *first,
                              float *second)
{
    if (!config || config->first.m != config->second.m ||
        config->first.k != config->second.k)
        return -22;
    int status = (config->first.m <= 512 ? run_matmul_4c<Format, true> : run_matmul_4c<Format>)(&config->first, a, first_bt, first);
    if (status == 0)
        status = (config->second.m <= 512 ? run_matmul_4c<Format, true> : run_matmul_4c<Format>)(&config->second, a, second_bt, second);
    return status;
}

template <typename Format>
static int run_mul_mat_id_4c(const DayanMulMatIdConfig *config,
                             const uint16_t *activation,
                             const uint16_t *weight, const int32_t *ids,
                             float *output)
{
    if (!mul_mat_id_config_valid_4c(config, sizeof(float)) || !activation ||
        !weight || !ids || !output)
        return -22;

    const DayanMatmulConfig matmul = moe_matmul_config_4c(
        config, sizeof(float));
    const uint64_t weight_elements = (uint64_t)config->n * config->k;
    int status = 0;

    for (uint32_t token = 0; token < config->tokens && status == 0; ++token) {
        for (uint32_t used = 0; used < config->experts_used && status == 0;
             ++used) {
            int32_t expert = 0;
            status = read_selected_expert_4c(
                config, ids, (uint64_t)token * config->experts_used + used,
                &expert);
            if (status == 0) {
                status = run_matmul_4c<Format, true>(
                    &matmul,
                    activation + moe_activation_offset_4c(config, token, used),
                    weight + (uint64_t)expert * weight_elements,
                    output + moe_output_offset_4c(config, token, used));
            }
        }
    }
    return status;
}

template <typename Format>
static int run_mul_mat_id_pair_4c(const DayanMulMatIdPairConfig *config,
                                  const uint16_t *activation,
                                  const uint16_t *first_weight,
                                  const uint16_t *second_weight,
                                  const int32_t *ids, float *first_output,
                                  float *second_output)
{
    if (!config ||
        !mul_mat_id_config_valid_4c(&config->first, sizeof(float)) ||
        !mul_mat_id_config_valid_4c(&config->second, sizeof(float)) ||
        !same_mul_mat_id_selection_4c(config->first, config->second) ||
        !activation || !first_weight || !second_weight || !ids ||
        !first_output || !second_output)
        return -22;

    const DayanMatmulConfig first_matmul = moe_matmul_config_4c(
        &config->first, sizeof(float));
    const DayanMatmulConfig second_matmul = moe_matmul_config_4c(
        &config->second, sizeof(float));
    const uint64_t first_weight_elements =
        (uint64_t)config->first.n * config->first.k;
    const uint64_t second_weight_elements =
        (uint64_t)config->second.n * config->second.k;
    int status = 0;

    for (uint32_t token = 0; token < config->first.tokens && status == 0;
         ++token) {
        for (uint32_t used = 0;
             used < config->first.experts_used && status == 0; ++used) {
            int32_t expert = 0;
            status = read_selected_expert_4c(
                &config->first, ids,
                (uint64_t)token * config->first.experts_used + used,
                &expert);
            const uint16_t *input = activation + moe_activation_offset_4c(
                &config->first, token, used);
            if (status == 0) {
                status = run_matmul_4c<Format, true>(
                    &first_matmul, input,
                    first_weight + (uint64_t)expert * first_weight_elements,
                    first_output + moe_output_offset_4c(
                        &config->first, token, used));
            }
            if (status == 0) {
                status = run_matmul_4c<Format, true>(
                    &second_matmul, input,
                    second_weight + (uint64_t)expert * second_weight_elements,
                    second_output + moe_output_offset_4c(
                        &config->second, token, used));
            }
        }
    }
    return status;
}

template <typename Format, bool GeGLU>
static int run_mul_mat_id_glu_4c(const DayanMulMatIdConfig *config,
                                 const uint16_t *activation,
                                 const uint16_t *gate_weight,
                                 const uint16_t *up_weight,
                                 const int32_t *ids, uint16_t *output)
{
    if (!mul_mat_id_config_valid_4c(config, sizeof(uint16_t)) ||
        !activation || !gate_weight || !up_weight || !ids || !output)
        return -22;

    DayanMatmulConfig epilogue = moe_matmul_config_4c(
        config, sizeof(uint16_t));
    const uint64_t weight_elements = (uint64_t)config->n * config->k;
    int status = 0;

    for (uint32_t token = 0; token < config->tokens && status == 0; ++token) {
        for (uint32_t used = 0; used < config->experts_used && status == 0;
             ++used) {
            int32_t expert = 0;
            status = read_selected_expert_4c(
                config, ids, (uint64_t)token * config->experts_used + used,
                &expert);
            const uint16_t *input = activation + moe_activation_offset_4c(
                config, token, used);
            const uint64_t weight_offset =
                (uint64_t)expert * weight_elements;
            if (status == 0) {
                status = run_matmul_glu_4c<Format, GeGLU, true>(
                    &epilogue, input, gate_weight + weight_offset, up_weight + weight_offset,
                    output + moe_output_offset_4c(config, token, used));
            }
        }
    }
    return status;
}

template <typename Format, bool HasGate>
static int postprocess_attention_projection_4c(
    const DayanAttentionProjectionConfig *config, const float *norm,
    const int32_t *positions, const uint16_t *raw_output, uint16_t *output)
{
    using namespace ggml_dayan_matmul_detail;

    attention_reset_scratch();
    attention_initialize_constants(config);
    attention_load_bytes(attention_uram(kAttentionNorm), norm,
                         config->head_dim * sizeof(float));
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    const uint32_t position_axes = config->rope_mode == 8U ? 4U : 1U;
    const uint32_t projection_step = config->head_dim * (HasGate ? 2U : 1U);
    const uint64_t jobs = (uint64_t)config->tokens * config->heads;

    for (uint64_t job = lane; job < jobs; job += Constants::kClusterCores) {
        const uint32_t token = (uint32_t)(job / config->heads);
        const uint32_t head = (uint32_t)(job % config->heads);
        attention_load_i32(attention_uram(kAttentionPositions),
                           positions + token, position_axes,
                           (uint64_t)config->tokens * sizeof(int32_t));
        const uint16_t *source = raw_output +
            (uint64_t)token * config->matmul.output_stride /
                sizeof(uint16_t) +
            (uint64_t)head * projection_step;
        uint16_t *destination = output +
            ((uint64_t)head * config->tokens + token) * config->head_dim;
        postprocess_attention_head<Format>(config, source, destination);
    }
    return finish_epilogue_4c();
}

template <typename Format, bool HasGate>
static int run_attention_projection_4c(
    const DayanAttentionProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, const float *norm, const int32_t *positions,
    uint16_t *raw_output, uint16_t *output)
{
    if (!ggml_dayan_matmul_detail::attention_projection_config_valid(
            config, HasGate) ||
        !activation || !weight || !norm || !positions || !raw_output ||
        !output)
        return -22;

    int status = (config->matmul.m <= 512 ? run_matmul_quantized_4c<Format, true> : run_matmul_quantized_4c<Format>)(&config->matmul, activation, weight, raw_output);
    if (status == 0)
        status = postprocess_attention_projection_4c<Format, HasGate>(
            config, norm, positions, raw_output, output);
    return status;
}

template <typename Format>
static int run_value_projection_4c(
    const DayanAttentionProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, uint16_t *raw_output, uint16_t *output)
{
    using namespace ggml_dayan_matmul_detail;

    if (!attention_projection_config_valid(config, false) || !activation ||
        !weight || !raw_output || !output)
        return -22;

    int status = (config->matmul.m <= 512 ? run_matmul_quantized_4c<Format, true> : run_matmul_quantized_4c<Format>)(&config->matmul, activation, weight, raw_output);
    if (status == 0) {
        const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
        const uint64_t jobs = (uint64_t)config->tokens * config->heads;
        for (uint64_t job = lane; job < jobs;
             job += Constants::kClusterCores) {
            const uint32_t token = (uint32_t)(job / config->heads);
            const uint32_t head = (uint32_t)(job % config->heads);
            const uint16_t *source = raw_output +
                (uint64_t)token * config->matmul.output_stride /
                    sizeof(uint16_t) +
                (uint64_t)head * config->head_dim;
            uint16_t *destination = output +
                ((uint64_t)head * config->tokens + token) * config->head_dim;
            attention_copy_bytes(destination, source,
                                 config->head_dim * sizeof(uint16_t));
        }
        status = finish_epilogue_4c();
    }
    return status;
}

template <typename Format>
static int run_k_cache_projection_4c(
    const DayanKVProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, const float *norm,
    const int32_t *positions_indices, uint16_t *raw_output, uint16_t *cache)
{
    using namespace ggml_dayan_matmul_detail;

    if (!kv_projection_config_valid(config) || !activation || !weight ||
        !norm || !positions_indices || !raw_output || !cache)
        return -22;

    const DayanAttentionProjectionConfig *projection = &config->projection;
    int status = (projection->matmul.m <= 512 ? run_matmul_quantized_4c<Format, true> : run_matmul_quantized_4c<Format>)(&projection->matmul, activation, weight, raw_output);
    if (status == 0) {
        attention_reset_scratch();
        attention_initialize_constants(projection);
        attention_load_bytes(attention_uram(kAttentionNorm), norm,
                             projection->head_dim * sizeof(float));
        const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
        const uint32_t position_axes = projection->rope_mode == 8U ? 4U : 1U;
        const int32_t *indices = positions_indices +
            (uint64_t)position_axes * projection->tokens;
        const uint32_t cache_width = projection->heads * projection->head_dim;
        for (uint32_t token = 0; token < projection->tokens && status == 0;
             ++token) {
            attention_load_i32(attention_uram(kAttentionPositions),
                               indices + token, 1U, sizeof(int32_t));
            const int32_t cache_row =
                *attention_uram_pointer<int32_t>(kAttentionPositions);
            if (cache_row < 0 || (uint32_t)cache_row >= config->cache_rows) {
                status = -22;
                break;
            }
            attention_load_i32(attention_uram(kAttentionPositions),
                               positions_indices + token, position_axes,
                               (uint64_t)projection->tokens * sizeof(int32_t));
            for (uint32_t head = lane; head < projection->heads;
                 head += Constants::kClusterCores) {
                const uint16_t *source = raw_output +
                    (uint64_t)token * projection->matmul.output_stride /
                        sizeof(uint16_t) +
                    (uint64_t)head * projection->head_dim;
                uint16_t *destination = cache +
                    (uint64_t)cache_row * cache_width +
                    (uint64_t)head * projection->head_dim;
                postprocess_attention_head<Format>(
                    projection, source, destination);
            }
            status = finish_epilogue_4c();
        }
    }
    return status;
}

template <typename Format>
static int run_v_cache_projection_4c(
    const DayanKVProjectionConfig *config, const uint16_t *activation,
    const uint16_t *weight, const int32_t *indices, uint16_t *raw_output,
    uint16_t *cache)
{
    using namespace ggml_dayan_matmul_detail;

    if (!kv_projection_config_valid(config) || !activation || !weight ||
        !indices || !raw_output || !cache)
        return -22;

    const DayanAttentionProjectionConfig *projection = &config->projection;
    int status = (projection->matmul.m <= 512 ? run_matmul_quantized_4c<Format, true> : run_matmul_quantized_4c<Format>)(&projection->matmul, activation, weight, raw_output);
    if (status == 0) {
        const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
        const uint32_t cache_width = projection->heads * projection->head_dim;
        for (uint32_t token = 0; token < projection->tokens && status == 0;
             ++token) {
            attention_load_i32(attention_uram(kAttentionPositions),
                               indices + token, 1U, sizeof(int32_t));
            const int32_t cache_row =
                *attention_uram_pointer<int32_t>(kAttentionPositions);
            if (cache_row < 0 || (uint32_t)cache_row >= config->cache_rows) {
                status = -22;
                break;
            }
            for (uint32_t head = lane; head < projection->heads;
                 head += Constants::kClusterCores) {
                const uint16_t *source = raw_output +
                    (uint64_t)token * projection->matmul.output_stride /
                        sizeof(uint16_t) +
                    (uint64_t)head * projection->head_dim;
                uint16_t *destination = cache +
                    (uint64_t)cache_row * cache_width +
                    (uint64_t)head * projection->head_dim;
                attention_copy_bytes(destination, source,
                                     projection->head_dim * sizeof(uint16_t));
            }
            status = finish_epilogue_4c();
        }
    }
    return status;
}

static inline void configure_gated_lowp_load_4c(
    const DayanGatedProjectionConfig *config)
{
    const uint32_t width = config->heads * config->head_dim;
    ggml_dayan_matmul_detail::configure_gated_packed_load<1, sizeof(uint16_t)>(
        config->head_dim, width,
        PostprocessLayout::kColumns * sizeof(uint16_t));
    ggml_dayan_matmul_detail::configure_gated_packed_to_f32(
        PostprocessLayout::kColumns * sizeof(uint16_t),
        PostprocessLayout::kColumns * sizeof(float));
}

static inline void configure_gated_f32_load_4c(
    const DayanGatedProjectionConfig *config)
{
    const uint32_t width = config->heads * config->head_dim;
    ggml_dayan_matmul_detail::configure_gated_packed_load<1, sizeof(float)>(
        config->head_dim, width,
        PostprocessLayout::kColumns * sizeof(float));
}

template <void (*Convert)()>
static inline void load_gated_lowp_tile_4c(
    const DayanGatedProjectionConfig *config, const void *gate,
    uint32_t row_base, uint32_t head_base, uint32_t heads, int rows,
    int columns)
{
    const uint32_t width = config->heads * config->head_dim;
    ggml_dayan_matmul_detail::configure_gated_packed_load_dims<1>(heads, rows);
    __tx_lsu_dma_d2u<1>(
        postprocess_uram(PostprocessLayout::kOutput),
        reinterpret_cast<uint64_t>(
            static_cast<const uint16_t *>(gate) +
            (uint64_t)row_base * 2U * width +
            (uint64_t)head_base * 2U * config->head_dim +
            config->head_dim));
    dayan_wait_lsu();
    ggml_dayan_matmul_detail::issue_gated_packed_to_f32<Convert>(
        postprocess_uram(PostprocessLayout::kInput1),
        postprocess_uram(PostprocessLayout::kOutput), rows, columns);
    dayan_wait_vec();
}

static inline void load_gated_f32_tile_4c(
    const DayanGatedProjectionConfig *config, const void *gate,
    uint32_t row_base, uint32_t head_base, uint32_t heads, int rows,
    int columns)
{
    (void) columns;
    const uint32_t width = config->heads * config->head_dim;
    ggml_dayan_matmul_detail::configure_gated_packed_load_dims<1>(heads, rows);
    __tx_lsu_dma_d2u<1>(
        postprocess_uram(PostprocessLayout::kInput1),
        reinterpret_cast<uint64_t>(
            static_cast<const float *>(gate) +
            (uint64_t)row_base * 2U * width +
            (uint64_t)head_base * 2U * config->head_dim +
            config->head_dim));
    dayan_wait_lsu();
}

template <typename Format,
          void (*ConfigureGate)(const DayanGatedProjectionConfig *),
          void (*LoadGate)(const DayanGatedProjectionConfig *, const void *,
                           uint32_t, uint32_t, uint32_t, int, int)>
static int gate_attention_input_4c(
    const DayanGatedProjectionConfig *config, const float *attention,
    const void *gate, uint16_t *output)
{
    using namespace ggml_dayan_matmul_detail;

    const uint32_t width = config->heads * config->head_dim;
    const uint32_t heads_per_tile =
        PostprocessLayout::kColumns / config->head_dim;
    const uint32_t row_tiles =
        (config->tokens + PostprocessLayout::kRows - 1U) /
        PostprocessLayout::kRows;
    const uint32_t head_tiles =
        (config->heads + heads_per_tile - 1U) / heads_per_tile;
    const uint64_t tile_count = (uint64_t)row_tiles * head_tiles;
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);

    initialize_postprocess_constants();
    configure_postprocess_load(0, (uint64_t)width * sizeof(float));
    configure_postprocess_store((uint64_t)width * sizeof(uint16_t));
    ConfigureGate(config);
    for (uint64_t tile = lane; tile < tile_count;
         tile += Constants::kClusterCores) {
        const uint32_t row_tile = (uint32_t)(tile / head_tiles);
        const uint32_t head_tile = (uint32_t)(tile % head_tiles);
        const uint32_t row_base = row_tile * PostprocessLayout::kRows;
        const uint32_t head_base = head_tile * heads_per_tile;
        const int rows = min_int(
            (int)(config->tokens - row_base),
            (int)PostprocessLayout::kRows);
        const uint32_t heads = (uint32_t)min_int(
            (int)(config->heads - head_base), (int)heads_per_tile);
        const int columns = heads * config->head_dim;

        load_postprocess_tile(
            0, postprocess_uram(PostprocessLayout::kInput0),
            attention + (uint64_t)row_base * width +
                (uint64_t)head_base * config->head_dim,
            rows, columns);
        LoadGate(config, gate, row_base, head_base, heads, rows, columns);
        ggml_dayan_vec_detail::issue_sigmoid_mul(
            {postprocess_uram(PostprocessLayout::kInput1),
             (long)PostprocessLayout::kColumns * (long)sizeof(float)},
            {postprocess_uram(PostprocessLayout::kInput0),
             (long)PostprocessLayout::kColumns * (long)sizeof(float)},
            {postprocess_uram(PostprocessLayout::kInput0),
             (long)PostprocessLayout::kColumns * (long)sizeof(float)},
            postprocess_constants(), rows, columns);
        dayan_wait_vec();
        const ggml_dayan_vec_detail::F32Tensor attention_tile = {
            postprocess_uram(PostprocessLayout::kInput0),
            (long)PostprocessLayout::kColumns * (long)sizeof(float)};
        ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
            attention_tile, attention_tile,
            {postprocess_uram(PostprocessLayout::kOutput),
             (long)PostprocessLayout::kColumns * (long)sizeof(uint16_t)},
            rows, columns);
        configure_postprocess_load(0, (uint64_t)width * sizeof(float));
        store_postprocess_tile(
            output + (uint64_t)row_base * width +
                (uint64_t)head_base * config->head_dim,
            rows, columns);
    }
    return finish_epilogue_4c();
}

template <typename Format,
          void (*ConfigureGate)(const DayanGatedProjectionConfig *),
          void (*LoadGate)(const DayanGatedProjectionConfig *, const void *,
                           uint32_t, uint32_t, uint32_t, int, int)>
static int run_gated_output_projection_impl_4c(
    const DayanGatedProjectionConfig *config, const float *attention,
    const void *gate, const uint16_t *weight, float *output, uint16_t *scratch)
{
    const uint64_t width = config ?
        (uint64_t)config->heads * config->head_dim : 0U;
    if (!config || !attention || !gate || !weight || !output || !scratch ||
        !config->tokens || !config->heads || !config->head_dim ||
        config->matmul.m != config->tokens || config->matmul.k != width ||
        config->head_dim > PostprocessLayout::kColumns ||
        config->head_dim % ggml_dayan_matmul_detail::Constants::kVecLanes !=
            0U ||
        config->matmul.m >= config->matmul.n)
        return -22;

    uint16_t *gated = scratch;
    int status = gate_attention_input_4c<Format, ConfigureGate, LoadGate>(
        config, attention, gate, gated);
    if (status == 0)
        status = (config->matmul.m <= 512 ? run_matmul_4c<Format, true> : run_matmul_4c<Format>)(&config->matmul, gated, weight, output);
    return status;
}

template <typename Format, void (*Convert)()>
static int run_gated_output_projection_4c(
    const DayanGatedProjectionConfig *config, const float *attention,
    const uint16_t *gate, const uint16_t *weight, float *output, uint16_t *scratch)
{
    return run_gated_output_projection_impl_4c<
        Format, configure_gated_lowp_load_4c,
        load_gated_lowp_tile_4c<Convert>>(
            config, attention, gate, weight, output, scratch);
}

template <typename Format>
static int run_gated_output_projection_f32_gate_4c(
    const DayanGatedProjectionConfig *config, const float *attention,
    const float *gate, const uint16_t *weight, float *output, uint16_t *scratch)
{
    return run_gated_output_projection_impl_4c<
        Format, configure_gated_f32_load_4c, load_gated_f32_tile_4c>(
            config, attention, gate, weight, output, scratch);
}

}  // namespace ggml_dayan_matmul_4c_detail

#endif
