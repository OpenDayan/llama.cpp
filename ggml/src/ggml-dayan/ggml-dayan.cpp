#include "ggml-dayan.h"

#include "ggml-backend-impl.h"
#include "ggml-dayan-graph.h"
#include "ggml-dayan-kernels.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

static constexpr uint32_t GGML_DAYAN_TIMEOUT_MS          = 3600000U;
static constexpr size_t   GGML_DAYAN_MATMUL_BUFFER_BYTES = 16U * 1024U * 1024U;
static constexpr uint64_t GGML_DAYAN_MATMUL_TILE_N       = 384U;

struct ggml_backend_dayan_context {
    npu_runtime_t * runtime = nullptr;
    npu_context_t * context = nullptr;
    npu_stream_t *  stream  = nullptr;
};

struct ggml_backend_dayan_buffers {
    npu_buffer_t * config  = nullptr;
    npu_buffer_t * weight0 = nullptr;
    npu_buffer_t * weight1 = nullptr;
    npu_buffer_t * input2  = nullptr;
    npu_buffer_t * input3  = nullptr;
    npu_buffer_t * output  = nullptr;
    npu_buffer_t * perf    = nullptr;
};

struct ggml_backend_dayan_activation_key {
    const ggml_tensor * tensor;
    enum ggml_type      type;

    bool operator==(const ggml_backend_dayan_activation_key & other) const {
        return tensor == other.tensor && type == other.type;
    }
};

struct ggml_backend_dayan_activation_key_hash {
    size_t operator()(const ggml_backend_dayan_activation_key & key) const {
        return std::hash<const ggml_tensor *>{}(key.tensor) ^ ((size_t) key.type << 1);
    }
};

struct ggml_backend_dayan_graph_buffers {
    std::vector<npu_buffer_t *> activations;
    std::unordered_map<ggml_backend_dayan_activation_key, npu_buffer_t *, ggml_backend_dayan_activation_key_hash>
        activation_by_tensor;
};

static int ggml_backend_dayan_event_wait(npu_event_t * event) {
    if (event == nullptr) {
        return -1;
    }

    npu_event_info_t info;
    const int        status         = npu_event_wait(event, GGML_DAYAN_TIMEOUT_MS, &info);
    const int        destroy_status = npu_event_destroy(event);

    if (status < 0) {
        return status;
    }
    if (info.status < std::numeric_limits<int>::min() || info.status > std::numeric_limits<int>::max()) {
        return -1;
    }
    if (info.status < 0) {
        return (int) info.status;
    }
    return destroy_status;
}

static int ggml_backend_dayan_buffer_create(npu_context_t * context, size_t size, npu_buffer_t ** buffer) {
    npu_buffer_desc_t desc;
    npu_buffer_desc_init(&desc, size);
    return npu_buffer_create(context, &desc, buffer);
}

static int ggml_backend_dayan_upload(npu_stream_t * stream, npu_buffer_t * buffer, const void * data, size_t size) {
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_h2d_async(stream, buffer, 0, data, size, &event);
    return status < 0 ? status : ggml_backend_dayan_event_wait(event);
}

static int ggml_backend_dayan_download(npu_stream_t * stream, void * data, npu_buffer_t * buffer, size_t size) {
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_d2h_async(stream, data, buffer, 0, size, &event);
    return status < 0 ? status : ggml_backend_dayan_event_wait(event);
}

static void ggml_backend_dayan_preserve_error(int candidate, int * status) {
    if (*status == 0 && candidate < 0) {
        *status = candidate;
    }
}

static void ggml_backend_dayan_buffers_free(ggml_backend_dayan_buffers * buffers, int * status) {
    if (buffers->perf != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->perf), status);
    }
    if (buffers->output != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->output), status);
    }
    if (buffers->input3 != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->input3), status);
    }
    if (buffers->input2 != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->input2), status);
    }
    if (buffers->weight1 != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->weight1), status);
    }
    if (buffers->weight0 != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->weight0), status);
    }
    if (buffers->config != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->config), status);
    }
}

static void ggml_backend_dayan_graph_buffers_free(ggml_backend_dayan_graph_buffers * buffers, int * status) {
    for (auto it = buffers->activations.rbegin(); it != buffers->activations.rend(); ++it) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(*it), status);
    }
}

static float ggml_backend_dayan_read_flash_value(const ggml_tensor * tensor, const char * address) {
    switch (tensor->type) {
        case GGML_TYPE_F32:
            {
                float value;
                std::memcpy(&value, address, sizeof(value));
                return value;
            }
        case GGML_TYPE_F16:
            {
                ggml_fp16_t value;
                std::memcpy(&value, address, sizeof(value));
                return ggml_fp16_to_fp32(value);
            }
        case GGML_TYPE_BF16:
            {
                ggml_bf16_t value;
                std::memcpy(&value, address, sizeof(value));
                return ggml_bf16_to_fp32(value);
            }
        default:
            GGML_ABORT("unsupported Flash-Attention input type");
    }
}

static bool ggml_backend_dayan_pack_flash_tensor(const ggml_tensor * tensor,
                                                 uint32_t            rows,
                                                 std::vector<ggml_bf16_t> * packed) {
    uint64_t count = (uint64_t) tensor->ne[0] * rows;
    count *= (uint64_t) tensor->ne[2];
    count *= (uint64_t) tensor->ne[3];
    if (rows == 0 || rows > (uint32_t) tensor->ne[1] || count > std::numeric_limits<size_t>::max()) {
        return false;
    }

    packed->resize((size_t) count);
    size_t target = 0;
    for (int64_t batch = 0; batch < tensor->ne[3]; ++batch) {
        for (int64_t head = 0; head < tensor->ne[2]; ++head) {
            for (uint32_t row = 0; row < rows; ++row) {
                for (int64_t dim = 0; dim < tensor->ne[0]; ++dim) {
                    const char * address = (const char *) tensor->data + batch * tensor->nb[3] +
                                           head * tensor->nb[2] + row * tensor->nb[1] + dim * tensor->nb[0];
                    (*packed)[target++] = ggml_fp32_to_bf16(ggml_backend_dayan_read_flash_value(tensor, address));
                }
            }
        }
    }
    return true;
}

static int ggml_backend_dayan_prepare_activation(ggml_backend_dayan_context *       ctx,
                                                 ggml_backend_dayan_graph_buffers * buffers,
                                                 const ggml_tensor *                activation,
                                                 enum ggml_type                     storage_type,
                                                 npu_buffer_t **                    result) {
    const ggml_backend_dayan_activation_key key    = { activation, storage_type };
    const auto                              cached = buffers->activation_by_tensor.find(key);
    if (cached != buffers->activation_by_tensor.end()) {
        *result = cached->second;
        return 0;
    }

    const int64_t  activation_count = ggml_nelements(activation);
    const size_t   activation_size  = (size_t) activation_count * sizeof(ggml_bf16_t);
    npu_buffer_t * device_buffer    = nullptr;
    int            status           = ggml_backend_dayan_buffer_create(ctx->context, activation_size, &device_buffer);

    if (status == 0) {
        buffers->activations.push_back(device_buffer);
    }

    if (status == 0 && activation->type == GGML_TYPE_F32 && storage_type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> activation_bf16((size_t) activation_count);
        ggml_fp32_to_bf16_row((const float *) activation->data, activation_bf16.data(), activation_count);
        status = ggml_backend_dayan_upload(ctx->stream, device_buffer, activation_bf16.data(), activation_size);
    } else if (status == 0 && activation->type == GGML_TYPE_F32 && storage_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> activation_f16((size_t) activation_count);
        ggml_fp32_to_fp16_row((const float *) activation->data, activation_f16.data(), activation_count);
        status = ggml_backend_dayan_upload(ctx->stream, device_buffer, activation_f16.data(), activation_size);
    } else if (status == 0 && activation->type == storage_type &&
               (storage_type == GGML_TYPE_BF16 || storage_type == GGML_TYPE_F16)) {
        status = ggml_backend_dayan_upload(ctx->stream, device_buffer, activation->data, activation_size);
    } else if (status == 0) {
        status = -1;
    }

    if (status == 0) {
        buffers->activation_by_tensor.emplace(key, device_buffer);
        *result = device_buffer;
    }
    return status;
}

static void ggml_backend_dayan_init_launch_attrs(ggml_backend_dayan_context * ctx,
                                                 const char *                 name,
                                                 npu_kernel_launch_attrs_t *  attrs) {
    npu_kernel_launch_attrs_init(attrs, ctx->context, ctx->stream, name);
    attrs->config.task_dim  = NPUTaskDim3{ 1, 1, 1 };
    attrs->config.task_mode = NPUTaskMode::kSingleCore;
    attrs->timeout_ms       = GGML_DAYAN_TIMEOUT_MS;
}

static bool ggml_backend_dayan_perf_valid(const DayanMatmulPerf & perf, uint64_t m, uint64_t n, uint64_t k) {
    return perf.magic == kDayanMatmulPerfMagic && perf.m == m && perf.n == n && perf.k == k &&
           perf.kernel_cycles != 0 && perf.pipeline_cycles != 0 && perf.pipeline_cycles <= perf.kernel_cycles;
}

static uint64_t ggml_backend_dayan_mul_mat_chunk_n(const ggml_tensor * weight, const ggml_tensor * dst) {
    const uint64_t n = weight->ne[1];
    if (ggml_nbytes(weight) <= GGML_DAYAN_MATMUL_BUFFER_BYTES && ggml_nbytes(dst) <= GGML_DAYAN_MATMUL_BUFFER_BYTES) {
        return n;
    }

    const uint64_t weight_rows = GGML_DAYAN_MATMUL_BUFFER_BYTES / weight->nb[1];
    const uint64_t output_rows = GGML_DAYAN_MATMUL_BUFFER_BYTES / (dst->ne[1] * sizeof(float));
    uint64_t       chunk_n     = std::min(n, std::min(weight_rows, output_rows));

    if (chunk_n >= GGML_DAYAN_MATMUL_TILE_N) {
        chunk_n -= chunk_n % GGML_DAYAN_MATMUL_TILE_N;
    } else {
        chunk_n -= chunk_n % 16U;
    }
    return chunk_n;
}

static int ggml_backend_dayan_mul_mat(ggml_backend_dayan_context * ctx,
                                      ggml_tensor *                dst,
                                      npu_buffer_t *               activation_buffer) {
    const ggml_tensor * weight     = dst->src[0];
    const ggml_tensor * activation = dst->src[1];

    const uint64_t m       = activation->ne[1];
    const uint64_t n       = weight->ne[1];
    const uint64_t k       = weight->ne[0];
    const uint64_t chunk_n = ggml_backend_dayan_mul_mat_chunk_n(weight, dst);

    DayanMatmulConfig          config = {};
    DayanMatmulPerf            perf   = {};
    ggml_backend_dayan_buffers buffers;
    std::vector<float>         output_chunk;
    uint64_t                   kernel_cycles   = 0;
    uint64_t                   pipeline_cycles = 0;
    uint64_t                   chunk_count     = 0;
    int                        status          = chunk_n == 0 ? -1 : 0;

    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, chunk_n * weight->nb[1], &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, m * chunk_n * sizeof(float), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(perf), &buffers.perf);
    }
    if (status == 0) {
        output_chunk.resize((size_t) (m * chunk_n));
    }

    for (uint64_t n_offset = 0; status == 0 && n_offset < n; n_offset += config.n) {
        config = {
            m,
            std::min(chunk_n, n - n_offset),
            k,
            k * sizeof(ggml_bf16_t),
            weight->nb[1],
            std::min(chunk_n, n - n_offset) * sizeof(float),
        };
        perf = {};

        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
        if (status == 0) {
            const void * weight_data = (const char *) weight->data + n_offset * weight->nb[1];
            status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, weight_data, config.n * weight->nb[1]);
        }
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            ggml_backend_dayan_init_launch_attrs(ctx, "ggml-mul-mat", &attrs);
            if (weight->type == GGML_TYPE_BF16) {
                status = matmul_bf16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.output,
                                        buffers.perf, &attrs);
            } else {
                status = matmul_fp16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.output,
                                        buffers.perf, &attrs);
            }
        }
        if (status == 0) {
            status = ggml_backend_dayan_download(ctx->stream, &perf, buffers.perf, sizeof(perf));
        }
        if (status == 0 && !ggml_backend_dayan_perf_valid(perf, m, config.n, k)) {
            GGML_LOG_ERROR("%s: invalid perf magic=%llx M=%llu N=%llu K=%llu kernel=%llu pipeline=%llu\n", __func__,
                           (unsigned long long) perf.magic, (unsigned long long) perf.m, (unsigned long long) perf.n,
                           (unsigned long long) perf.k, (unsigned long long) perf.kernel_cycles,
                           (unsigned long long) perf.pipeline_cycles);
            status = -1;
        }
        if (status == 0) {
            status = ggml_backend_dayan_download(ctx->stream, output_chunk.data(), buffers.output,
                                                 m * config.n * sizeof(float));
        }
        if (status == 0) {
            for (uint64_t row = 0; row < m; ++row) {
                std::memcpy((char *) dst->data + row * dst->nb[1] + n_offset * sizeof(float),
                            output_chunk.data() + row * config.n, config.n * sizeof(float));
            }
            kernel_cycles += perf.kernel_cycles;
            pipeline_cycles += perf.pipeline_cycles;
            ++chunk_count;
        }
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan matrix multiplication failed with status %d\n", __func__, status);
    } else {
        GGML_LOG_DEBUG("%s: M=%llu N=%llu K=%llu, chunks=%llu, kernel_cycles=%llu, pipeline_cycles=%llu\n", __func__,
                       (unsigned long long) m, (unsigned long long) n, (unsigned long long) k,
                       (unsigned long long) chunk_count, (unsigned long long) kernel_cycles,
                       (unsigned long long) pipeline_cycles);
    }
    return status;
}

static int ggml_backend_dayan_mul_mat_quantized(ggml_backend_dayan_context *          ctx,
                                                const ggml_backend_dayan_graph_slot & slot,
                                                npu_buffer_t *                        activation_buffer) {
    const ggml_tensor * mul_mat    = slot.nodes[0];
    const ggml_tensor * weight     = mul_mat->src[0];
    const ggml_tensor * activation = mul_mat->src[1];
    ggml_tensor *       output     = slot.nodes[1];

    const uint64_t m = activation->ne[1];
    const uint64_t n = weight->ne[1];
    const uint64_t k = weight->ne[0];

    const DayanMatmulConfig config = {
        m, n, k, k * sizeof(uint16_t), weight->nb[1], output->nb[1],
    };
    DayanMatmulPerf            perf = {};
    ggml_backend_dayan_buffers buffers;
    int                        status = 0;

    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(weight), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(output), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(perf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, weight->data, ggml_nbytes(weight));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-mm-lowp", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     matmul_quantized_bf16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.output,
                                              buffers.perf, &attrs) :
                     matmul_quantized_fp16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.output,
                                              buffers.perf, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, &perf, buffers.perf, sizeof(perf));
    }
    if (status == 0 && !ggml_backend_dayan_perf_valid(perf, m, n, k)) {
        status = -1;
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output->data, buffers.output, ggml_nbytes(output));
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan quantized matrix multiplication failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_mul_mat_glu(ggml_backend_dayan_context *          ctx,
                                          const ggml_backend_dayan_graph_slot & slot,
                                          npu_buffer_t *                        activation_buffer) {
    const ggml_tensor * gate       = slot.nodes[0];
    const ggml_tensor * up         = slot.nodes[1];
    ggml_tensor *       output     = slot.nodes[3];
    const ggml_tensor * activation = gate->src[1];

    const uint64_t          m      = activation->ne[1];
    const uint64_t          n      = gate->src[0]->ne[1];
    const uint64_t          k      = gate->src[0]->ne[0];
    const DayanMatmulConfig config = {
        m, n, k, k * sizeof(uint16_t), gate->src[0]->nb[1], output->nb[1],
    };
    DayanMatmulPerf            perf = {};
    ggml_backend_dayan_buffers buffers;
    int                        status = 0;

    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(gate->src[0]), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(up->src[0]), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(output), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(perf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, gate->src[0]->data, ggml_nbytes(gate->src[0]));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, up->src[0]->data, ggml_nbytes(up->src[0]));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(
            ctx, slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ? "ggml-swiglu" : "ggml-geglu", &attrs);

        if (gate->src[0]->type == GGML_TYPE_BF16) {
            status = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ?
                         matmul_swiglu_bf16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.weight1,
                                               buffers.output, buffers.perf, &attrs) :
                         matmul_geglu_bf16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.weight1,
                                              buffers.output, buffers.perf, &attrs);
        } else {
            status = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ?
                         matmul_swiglu_fp16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.weight1,
                                               buffers.output, buffers.perf, &attrs) :
                         matmul_geglu_fp16_1c(buffers.config, activation_buffer, buffers.weight0, buffers.weight1,
                                              buffers.output, buffers.perf, &attrs);
        }
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, &perf, buffers.perf, sizeof(perf));
    }
    if (status == 0 && !ggml_backend_dayan_perf_valid(perf, m, n, k)) {
        status = -1;
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output->data, buffers.output, ggml_nbytes(output));
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan fused matrix multiplication failed with status %d\n", __func__, status);
    } else {
        GGML_LOG_DEBUG("%s: M=%llu N=%llu K=%llu, kernel_cycles=%llu, pipeline_cycles=%llu\n", __func__,
                       (unsigned long long) m, (unsigned long long) n, (unsigned long long) k,
                       (unsigned long long) perf.kernel_cycles, (unsigned long long) perf.pipeline_cycles);
    }
    return status;
}

static int ggml_backend_dayan_flash_attention(ggml_backend_dayan_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    float               scale;
    std::memcpy(&scale, dst->op_params, sizeof(scale));

    uint32_t causal_offset = 0;
    uint32_t kv_len        = 0;
    if (!ggml_backend_dayan_flash_attention_lengths(dst, &causal_offset, &kv_len)) {
        GGML_LOG_ERROR("%s: unsupported Flash-Attention mask\n", __func__);
        return -1;
    }

    DayanFlashAttentionConfig config = {
        (uint32_t) q->ne[3],
        (uint32_t) q->ne[2],
        (uint32_t) k->ne[2],
        (uint32_t) q->ne[1],
        kv_len,
        (uint32_t) q->ne[0],
        0,
        causal_offset,
        0,
    };
    std::memcpy(&config.scale_bits, &scale, sizeof(scale));
    if (dst->src[3] != nullptr) {
        config.flags = 1U;
    }

    std::vector<ggml_bf16_t> packed_q;
    std::vector<ggml_bf16_t> packed_k;
    std::vector<ggml_bf16_t> packed_v;
    if (!ggml_backend_dayan_pack_flash_tensor(q, config.q_len, &packed_q) ||
        !ggml_backend_dayan_pack_flash_tensor(k, config.kv_len, &packed_k) ||
        !ggml_backend_dayan_pack_flash_tensor(v, config.kv_len, &packed_v)) {
        GGML_LOG_ERROR("%s: cannot pack Flash-Attention inputs\n", __func__);
        return -1;
    }

    const size_t               output_size = ggml_nbytes(dst);
    ggml_backend_dayan_buffers buffers;
    int64_t                    device_status = -22;
    int                        status        = 0;

    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, packed_q.size() * sizeof(packed_q[0]), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, packed_k.size() * sizeof(packed_k[0]), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, packed_v.size() * sizeof(packed_v[0]), &buffers.input2);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_size, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(device_status), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, packed_q.data(),
                                           packed_q.size() * sizeof(packed_q[0]));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, packed_k.data(),
                                           packed_k.size() * sizeof(packed_k[0]));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input2, packed_v.data(),
                                           packed_v.size() * sizeof(packed_v[0]));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-fattn", &attrs);
        status = flash_attention_bf16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                         buffers.output, buffers.perf, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, &device_status, buffers.perf, sizeof(device_status));
    }
    if (status == 0 && device_status < 0) {
        status = (int) device_status;
    }

    std::vector<float> reference_layout;
    if (status == 0) {
        reference_layout.resize((size_t) ggml_nelements(dst));
        status = ggml_backend_dayan_download(ctx->stream, reference_layout.data(), buffers.output, output_size);
    }
    if (status == 0) {
        float * output = (float *) dst->data;
        for (uint32_t batch = 0; batch < config.batch; ++batch) {
            for (uint32_t query = 0; query < config.q_len; ++query) {
                for (uint32_t head = 0; head < config.q_heads; ++head) {
                    const size_t source =
                        (((size_t) batch * config.q_heads + head) * config.q_len + query) * config.head_dim;
                    const size_t target =
                        (((size_t) batch * config.q_len + query) * config.q_heads + head) * config.head_dim;
                    std::memcpy(output + target, reference_layout.data() + source, config.head_dim * sizeof(float));
                }
            }
        }
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan Flash-Attention failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_gated_output_projection(ggml_backend_dayan_context *          ctx,
                                                      const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * sigmoid    = slot.nodes[0];
    const ggml_tensor * gated      = slot.nodes[1];
    ggml_tensor *       projection = slot.nodes[2];
    const ggml_tensor * qg         = slot.nodes[3];
    const ggml_tensor * gate_view  = sigmoid->src[0]->src[0];
    const ggml_tensor * attention  = gated->src[0] == sigmoid ? gated->src[1] : gated->src[0];
    const ggml_tensor * weight     = projection->src[0];
    const uint64_t      tokens     = attention->ne[1];
    const uint64_t      heads      = gate_view->ne[1];
    const uint64_t      head_dim   = gate_view->ne[0];
    const uint64_t      width      = heads * head_dim;

    const DayanGatedProjectionConfig config = {
        { tokens, (uint64_t) weight->ne[1], width, width * sizeof(uint16_t), weight->nb[1], projection->nb[1] },
        (uint32_t) tokens,
        (uint32_t) heads,
        (uint32_t) head_dim,
    };
    DayanMatmulPerf       perf = {};
    std::vector<uint16_t> qg_lowp((size_t) ggml_nelements(qg));
    if (weight->type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row((const float *) qg->data, (ggml_bf16_t *) qg_lowp.data(), ggml_nelements(qg));
    } else {
        ggml_fp32_to_fp16_row((const float *) qg->data, (ggml_fp16_t *) qg_lowp.data(), ggml_nelements(qg));
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(attention), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, qg_lowp.size() * sizeof(uint16_t), &buffers.weight1);
    }
    if (status == 0) {
        status =
            ggml_backend_dayan_buffer_create(ctx->context, (size_t) tokens * width * sizeof(uint16_t), &buffers.input2);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(weight), &buffers.input3);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(projection), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(perf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, attention->data, ggml_nbytes(attention));
    }
    if (status == 0) {
        status =
            ggml_backend_dayan_upload(ctx->stream, buffers.weight1, qg_lowp.data(), qg_lowp.size() * sizeof(uint16_t));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input3, weight->data, ggml_nbytes(weight));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "gate-out", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     project_gated_output_bf16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                                  buffers.input3, buffers.output, buffers.perf, &attrs) :
                     project_gated_output_fp16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                                  buffers.input3, buffers.output, buffers.perf, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, &perf, buffers.perf, sizeof(perf));
    }
    if (status == 0 && !ggml_backend_dayan_perf_valid(perf, tokens, weight->ne[1], width)) {
        status = -1;
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, projection->data, buffers.output, ggml_nbytes(projection));
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan gated output projection failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_ssm_conv(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * conv   = slot.nodes[0];
    const ggml_tensor * input  = conv->src[0];
    const ggml_tensor * weight = conv->src[1];
    ggml_tensor * output = slot.kind == ggml_backend_dayan_slot_kind::ssm_conv_silu ? slot.nodes[1] : slot.nodes[0];
    const DayanSSMConvConfig config = {
        (uint32_t) input->ne[0],  (uint32_t) input->ne[1],
        (uint32_t) output->ne[1], (uint32_t) input->ne[2],
        (uint32_t) weight->ne[0], slot.kind == ggml_backend_dayan_slot_kind::ssm_conv_silu ? 1U : 0U,
    };
    DayanSSMConvPerf           perf = {};
    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(input), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(weight), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(output), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(perf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input->data, ggml_nbytes(input));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, weight->data, ggml_nbytes(weight));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ssm-conv", &attrs);
        status =
            ssm_conv_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output, buffers.perf, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, &perf, buffers.perf, sizeof(perf));
    }
    if (status == 0 && (perf.magic != kDayanSSMConvPerfMagic || perf.cycles == 0)) {
        status = -1;
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output->data, buffers.output, ggml_nbytes(output));
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan SSM convolution failed with status %d\n", __func__, status);
    }
    return status;
}

static const char * ggml_backend_dayan_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_DAYAN_NAME;
}

static void ggml_backend_dayan_context_free(ggml_backend_dayan_context * ctx) {
    if (ctx->stream != nullptr) {
        npu_stream_destroy(ctx->stream);
    }
    if (ctx->context != nullptr) {
        npu_context_destroy(ctx->context);
    }
    if (ctx->runtime != nullptr) {
        npu_runtime_close(ctx->runtime);
    }
    delete ctx;
}

static void ggml_backend_dayan_free(ggml_backend_t backend) {
    ggml_backend_dayan_context_free((ggml_backend_dayan_context *) backend->context);
    delete backend;
}

static enum ggml_status ggml_backend_dayan_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_dayan_context *  ctx = (ggml_backend_dayan_context *) backend->context;
    ggml_backend_dayan_graph_plan plan;
    const ggml_tensor *           unsupported = nullptr;

    if (!ggml_backend_dayan_compile_graph(cgraph, &plan, &unsupported)) {
        GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(unsupported));
        return GGML_STATUS_FAILED;
    }

    ggml_backend_dayan_graph_buffers graph_buffers;
    int                              status = 0;

    for (const ggml_backend_dayan_graph_slot & slot : plan.slots) {
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::mul_mat:
            case ggml_backend_dayan_slot_kind::mul_mat_pair:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    for (size_t i = 0; status == 0 && i < slot.node_count; ++i) {
                        status = ggml_backend_dayan_mul_mat(ctx, slot.nodes[i], activation_buffer);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::mul_mat_quantized:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat_quantized(ctx, slot, activation_buffer);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::mul_mat_swiglu:
            case ggml_backend_dayan_slot_kind::mul_mat_geglu:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat_glu(ctx, slot, activation_buffer);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::flash_attention:
                status = ggml_backend_dayan_flash_attention(ctx, slot.nodes[0]);
                break;
            case ggml_backend_dayan_slot_kind::gated_output_projection:
                status = ggml_backend_dayan_gated_output_projection(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::ssm_conv:
            case ggml_backend_dayan_slot_kind::ssm_conv_silu:
                status = ggml_backend_dayan_ssm_conv(ctx, slot);
                break;
        }
        if (status < 0) {
            break;
        }
    }

    ggml_backend_dayan_graph_buffers_free(&graph_buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: graph execution failed with status %d\n", __func__, status);
        return GGML_STATUS_FAILED;
    }

    if (!plan.slots.empty()) {
        GGML_LOG_DEBUG(
            "%s: lowered %d nodes to %zu slots, paired=%zu, fused=%zu, attention=%zu, gated=%zu, conv=%zu, "
            "activations=%zu\n",
            __func__, cgraph->n_nodes, plan.slots.size(), plan.paired_mul_mat_slots, plan.fused_mul_mat_slots,
            plan.flash_attention_slots, plan.gated_output_slots, plan.ssm_conv_slots,
            graph_buffers.activation_by_tensor.size());
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_dayan_i = {
    /* .get_name                = */ ggml_backend_dayan_get_name,
    /* .free                    = */ ggml_backend_dayan_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_dayan_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_dayan_guid(void) {
    static ggml_guid guid = { 0x8d, 0x1d, 0xd5, 0x57, 0x97, 0xe0, 0x4d, 0x45,
                              0x95, 0x8d, 0x61, 0x79, 0x61, 0x6e, 0x00, 0x01 };
    return &guid;
}

ggml_backend_t ggml_backend_dayan_init(void) {
    ggml_backend_dayan_context * ctx = new ggml_backend_dayan_context;
    npu_runtime_options_t        options;
    npu_runtime_options_init(&options);
    options.default_timeout_ms = GGML_DAYAN_TIMEOUT_MS;

    const char * socket = std::getenv("DAYAN_SERVER_SOCKET");
    if (socket != nullptr && socket[0] != '\0') {
        options.server_socket = socket;
    }

    int status = npu_runtime_open(&ctx->runtime, &options);
    if (status == 0) {
        status = npu_context_create(ctx->runtime, nullptr, &ctx->context);
    }
    if (status == 0) {
        status = npu_stream_create(ctx->context, nullptr, &ctx->stream);
    }
    if (status < 0) {
        GGML_LOG_ERROR("%s: failed to initialize the Dayan runtime with status %d\n", __func__, status);
        ggml_backend_dayan_context_free(ctx);
        return nullptr;
    }

    return new ggml_backend{
        /* .guid    = */ ggml_backend_dayan_guid(),
        /* .iface   = */ ggml_backend_dayan_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_dayan_reg(), 0),
        /* .context = */ ctx,
    };
}

bool ggml_backend_is_dayan(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_dayan_guid());
}

static const char * ggml_backend_dayan_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_DAYAN_NAME;
}

static const char * ggml_backend_dayan_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Dayan RISC-V NPU";
}

static void ggml_backend_dayan_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_dayan_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_dayan_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_dayan_device_get_name(dev);
    props->description = ggml_backend_dayan_device_get_description(dev);
    props->type        = ggml_backend_dayan_device_get_type(dev);
    ggml_backend_dayan_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_dayan_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_dayan_init();
}

static ggml_backend_buffer_type_t ggml_backend_dayan_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_dayan_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                                            void *             ptr,
                                                                            size_t             size,
                                                                            size_t             max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_dayan_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_CONT:
            return ggml_backend_dayan_supports_gate_cont(op);
        case GGML_OP_MUL_MAT:
            return ggml_backend_dayan_supports_mul_mat(op);
        case GGML_OP_GLU:
            return ggml_backend_dayan_supports_glu(op);
        case GGML_OP_CPY:
            return ggml_backend_dayan_supports_quantized_glu(op) || ggml_backend_dayan_supports_quantized_mul_mat(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_backend_dayan_supports_flash_attention(op);
        case GGML_OP_UNARY:
            return ggml_backend_dayan_supports_gate_sigmoid(op) || ggml_backend_dayan_supports_ssm_conv_silu(op);
        case GGML_OP_MUL:
            return ggml_backend_dayan_supports_gated_attention(op);
        case GGML_OP_SSM_CONV:
            return ggml_backend_dayan_supports_ssm_conv(op);
        default:
            return false;
    }
}

static bool ggml_backend_dayan_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static bool ggml_backend_dayan_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_dayan_device_supports_op(dev, op);
}

static const ggml_backend_device_i ggml_backend_dayan_device_i = {
    /* .get_name             = */ ggml_backend_dayan_device_get_name,
    /* .get_description      = */ ggml_backend_dayan_device_get_description,
    /* .get_memory           = */ ggml_backend_dayan_device_get_memory,
    /* .get_type             = */ ggml_backend_dayan_device_get_type,
    /* .get_props            = */ ggml_backend_dayan_device_get_props,
    /* .init_backend         = */ ggml_backend_dayan_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_dayan_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_dayan_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_dayan_device_supports_op,
    /* .supports_buft        = */ ggml_backend_dayan_device_supports_buft,
    /* .offload_op           = */ ggml_backend_dayan_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_backend_dayan_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_DAYAN_NAME;
}

static size_t ggml_backend_dayan_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_dayan_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_dayan_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

static void * ggml_backend_dayan_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_graph_rewrite") == 0) {
        return (void *) ggml_backend_dayan_graph_rewrite;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_dayan_reg_i = {
    /* .get_name         = */ ggml_backend_dayan_reg_get_name,
    /* .get_device_count = */ ggml_backend_dayan_reg_get_device_count,
    /* .get_device       = */ ggml_backend_dayan_reg_get_device,
    /* .get_proc_address = */ ggml_backend_dayan_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_dayan_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_dayan_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_dayan_reg)
