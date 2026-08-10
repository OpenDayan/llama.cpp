#include "ggml-dayan.h"

#include "ggml-backend-impl.h"
#include "ggml-dayan-graph.h"
#include "ggml-dayan-kernels.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
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
    npu_buffer_t * output1 = nullptr;
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

static int ggml_backend_dayan_download_at(npu_stream_t * stream,
                                          void *         data,
                                          npu_buffer_t * buffer,
                                          uint64_t       offset,
                                          size_t         size) {
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_d2h_async(stream, data, buffer, offset, size, &event);
    return status < 0 ? status : ggml_backend_dayan_event_wait(event);
}

static int ggml_backend_dayan_copy(npu_stream_t * stream,
                                   npu_buffer_t * destination,
                                   uint64_t       destination_offset,
                                   npu_buffer_t * source,
                                   uint64_t       source_offset,
                                   size_t         size) {
    npu_event_t * event  = nullptr;
    const int     status =
        npu_memcpy_d2d_async(stream, destination, destination_offset, source, source_offset, size, &event);
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
    if (buffers->output1 != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->output1), status);
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

static int ggml_backend_dayan_register_activation(ggml_backend_dayan_graph_buffers * buffers,
                                                  const ggml_tensor *                tensor,
                                                  enum ggml_type                     type,
                                                  npu_buffer_t **                    buffer) {
    if (buffer == nullptr || *buffer == nullptr) {
        return -1;
    }

    const ggml_backend_dayan_activation_key key = { tensor, type };
    if (!buffers->activation_by_tensor.emplace(key, *buffer).second) {
        return -1;
    }
    buffers->activations.push_back(*buffer);
    *buffer = nullptr;
    return 0;
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

static bool ggml_backend_dayan_pack_f32(const ggml_tensor * tensor, std::vector<float> * packed) {
    if (tensor == nullptr || packed == nullptr || tensor->data == nullptr || tensor->type != GGML_TYPE_F32 ||
        ggml_nelements(tensor) <= 0) {
        return false;
    }

    packed->resize((size_t) ggml_nelements(tensor));
    size_t target = 0;
    for (int64_t index3 = 0; index3 < tensor->ne[3]; ++index3) {
        for (int64_t index2 = 0; index2 < tensor->ne[2]; ++index2) {
            for (int64_t index1 = 0; index1 < tensor->ne[1]; ++index1) {
                for (int64_t index0 = 0; index0 < tensor->ne[0]; ++index0) {
                    const char * address = (const char *) tensor->data + index3 * tensor->nb[3] +
                                           index2 * tensor->nb[2] + index1 * tensor->nb[1] +
                                           index0 * tensor->nb[0];
                    std::memcpy(&(*packed)[target++], address, sizeof(float));
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_dayan_pack_repeated_f32(const ggml_tensor * source,
                                                 const ggml_tensor * shape,
                                                 std::vector<float> * packed) {
    if (source == nullptr || shape == nullptr || packed == nullptr || source->data == nullptr ||
        source->type != GGML_TYPE_F32 || !ggml_can_repeat(source, shape) || ggml_nelements(shape) <= 0) {
        return false;
    }

    packed->resize((size_t) ggml_nelements(shape));
    size_t target = 0;
    for (int64_t index3 = 0; index3 < shape->ne[3]; ++index3) {
        for (int64_t index2 = 0; index2 < shape->ne[2]; ++index2) {
            for (int64_t index1 = 0; index1 < shape->ne[1]; ++index1) {
                for (int64_t index0 = 0; index0 < shape->ne[0]; ++index0) {
                    const int64_t source3 = index3 % source->ne[3];
                    const int64_t source2 = index2 % source->ne[2];
                    const int64_t source1 = index1 % source->ne[1];
                    const int64_t source0 = index0 % source->ne[0];
                    const char * address = (const char *) source->data + source3 * source->nb[3] +
                                           source2 * source->nb[2] + source1 * source->nb[1] +
                                           source0 * source->nb[0];
                    std::memcpy(&(*packed)[target++], address, sizeof(float));
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_dayan_unpack_f32(ggml_tensor * tensor, const std::vector<float> & packed) {
    if (tensor == nullptr || tensor->data == nullptr || tensor->type != GGML_TYPE_F32 ||
        packed.size() != (size_t) ggml_nelements(tensor)) {
        return false;
    }

    size_t source = 0;
    for (int64_t index3 = 0; index3 < tensor->ne[3]; ++index3) {
        for (int64_t index2 = 0; index2 < tensor->ne[2]; ++index2) {
            for (int64_t index1 = 0; index1 < tensor->ne[1]; ++index1) {
                for (int64_t index0 = 0; index0 < tensor->ne[0]; ++index0) {
                    char * address = (char *) tensor->data + index3 * tensor->nb[3] + index2 * tensor->nb[2] +
                                     index1 * tensor->nb[1] + index0 * tensor->nb[0];
                    std::memcpy(address, &packed[source++], sizeof(float));
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_dayan_type_code(enum ggml_type type, uint32_t * code) {
    if (code == nullptr) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_F32:
            *code = kDayanDataTypeF32;
            return true;
        case GGML_TYPE_BF16:
            *code = kDayanDataTypeBF16;
            return true;
        case GGML_TYPE_F16:
            *code = kDayanDataTypeF16;
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_dayan_pack_scalar(const ggml_tensor * tensor, std::vector<uint8_t> * packed) {
    if (tensor == nullptr || packed == nullptr || tensor->data == nullptr || ggml_blck_size(tensor->type) != 1 ||
        ggml_nelements(tensor) <= 0) {
        return false;
    }

    const size_t element_size = ggml_type_size(tensor->type);
    packed->resize((size_t) ggml_nelements(tensor) * element_size);
    size_t target = 0;
    for (int64_t index3 = 0; index3 < tensor->ne[3]; ++index3) {
        for (int64_t index2 = 0; index2 < tensor->ne[2]; ++index2) {
            for (int64_t index1 = 0; index1 < tensor->ne[1]; ++index1) {
                for (int64_t index0 = 0; index0 < tensor->ne[0]; ++index0) {
                    const char * address = (const char *) tensor->data + index3 * tensor->nb[3] +
                                           index2 * tensor->nb[2] + index1 * tensor->nb[1] +
                                           index0 * tensor->nb[0];
                    std::memcpy(packed->data() + target, address, element_size);
                    target += element_size;
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_dayan_unpack_scalar(ggml_tensor * tensor, const std::vector<uint8_t> & packed) {
    if (tensor == nullptr || tensor->data == nullptr || ggml_blck_size(tensor->type) != 1 ||
        ggml_nelements(tensor) <= 0) {
        return false;
    }

    const size_t element_size = ggml_type_size(tensor->type);
    if (packed.size() != (size_t) ggml_nelements(tensor) * element_size) {
        return false;
    }
    size_t source = 0;
    for (int64_t index3 = 0; index3 < tensor->ne[3]; ++index3) {
        for (int64_t index2 = 0; index2 < tensor->ne[2]; ++index2) {
            for (int64_t index1 = 0; index1 < tensor->ne[1]; ++index1) {
                for (int64_t index0 = 0; index0 < tensor->ne[0]; ++index0) {
                    char * address = (char *) tensor->data + index3 * tensor->nb[3] + index2 * tensor->nb[2] +
                                     index1 * tensor->nb[1] + index0 * tensor->nb[0];
                    std::memcpy(address, packed.data() + source, element_size);
                    source += element_size;
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_dayan_read_index(const ggml_tensor * indices,
                                          int64_t             index0,
                                          int64_t             index1,
                                          int64_t             index2,
                                          uint64_t *          value) {
    if (indices == nullptr || value == nullptr || indices->data == nullptr) {
        return false;
    }
    const char * address = (const char *) indices->data + index0 * indices->nb[0] + index1 * indices->nb[1] +
                           index2 * indices->nb[2];
    if (indices->type == GGML_TYPE_I32) {
        int32_t index;
        std::memcpy(&index, address, sizeof(index));
        if (index < 0) {
            return false;
        }
        *value = (uint64_t) index;
        return true;
    }
    if (indices->type == GGML_TYPE_I64) {
        int64_t index;
        std::memcpy(&index, address, sizeof(index));
        if (index < 0) {
            return false;
        }
        *value = (uint64_t) index;
        return true;
    }
    return false;
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

static bool ggml_backend_dayan_append_gdn_tensor(const ggml_tensor *  tensor,
                                                 uint32_t             heads,
                                                 uint32_t             tokens,
                                                 uint32_t             sequences,
                                                 std::vector<float> * input) {
    if (tensor->ne[3] <= 0 || sequences % (uint32_t) tensor->ne[3] != 0) {
        return false;
    }

    const uint32_t sequence_repeat = sequences / (uint32_t) tensor->ne[3];
    const size_t   row_size        = (size_t) tensor->ne[0] * sizeof(float);
    const size_t   old_size        = input->size();
    const uint64_t added_values    = (uint64_t) sequences * tokens * heads * tensor->ne[0];
    if (added_values > std::numeric_limits<size_t>::max() - old_size) {
        return false;
    }
    input->resize(old_size + (size_t) added_values);

    size_t target = old_size;
    for (uint32_t sequence = 0; sequence < sequences; ++sequence) {
        const uint32_t source_sequence = sequence / sequence_repeat;
        for (uint32_t token = 0; token < tokens; ++token) {
            for (uint32_t head = 0; head < heads; ++head) {
                const uint32_t source_head = head % (uint32_t) tensor->ne[1];
                const char *   source      = (const char *) tensor->data + source_sequence * tensor->nb[3] +
                                             token * tensor->nb[2] + source_head * tensor->nb[1];
                std::memcpy(input->data() + target, source, row_size);
                target += (size_t) tensor->ne[0];
            }
        }
    }
    return true;
}

static void ggml_backend_dayan_append_contiguous_f32(const ggml_tensor * tensor, std::vector<float> * input) {
    const size_t old_size = input->size();
    input->resize(old_size + (size_t) ggml_nelements(tensor));
    std::memcpy(input->data() + old_size, tensor->data, ggml_nbytes(tensor));
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

static DayanNormConfig ggml_backend_dayan_norm_config(const ggml_tensor * rms) {
    DayanNormConfig config = {
        (uint32_t) rms->ne[0],
        (uint32_t) ggml_nrows(rms),
        0,
        0,
    };
    std::memcpy(&config.epsilon_bits, rms->op_params, sizeof(config.epsilon_bits));
    return config;
}

static const ggml_tensor * ggml_backend_dayan_norm_scale(const ggml_tensor * mul, const ggml_tensor * rms) {
    return mul->src[0] == rms ? mul->src[1] : mul->src[0];
}

static int ggml_backend_dayan_elementwise_binary(ggml_backend_dayan_context *          ctx,
                                                 const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * left   = output->src[0];
    const ggml_tensor * right  = output->src[1];
    const DayanElementwiseConfig config = { (uint64_t) ggml_nelements(output) };
    std::vector<float> left_values;
    std::vector<float> right_values;
    std::vector<float> output_values((size_t) config.elements);
    if (!ggml_backend_dayan_pack_f32(left, &left_values) ||
        !ggml_backend_dayan_pack_repeated_f32(right, output, &right_values)) {
        return -1;
    }

    const size_t bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, left_values.data(), bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, right_values.data(), bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::elementwise_add:
                ggml_backend_dayan_init_launch_attrs(ctx, "elementwise-add", &attrs);
                status = elementwise_add_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output,
                                                &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_mul:
                ggml_backend_dayan_init_launch_attrs(ctx, "elementwise-mul", &attrs);
                status = elementwise_mul_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output,
                                                &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_sub:
                ggml_backend_dayan_init_launch_attrs(ctx, "elementwise-sub", &attrs);
                status = elementwise_sub_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output,
                                                &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_div:
                ggml_backend_dayan_init_launch_attrs(ctx, "elementwise-div", &attrs);
                status = elementwise_div_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output,
                                                &attrs);
                break;
            default:
                status = -1;
                break;
        }
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_scale(ggml_backend_dayan_context *          ctx,
                                    const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanScaleConfig    config = { (uint64_t) ggml_nelements(output), 0, 0 };
    std::memcpy(&config.scale_bits, output->op_params, sizeof(config.scale_bits));
    std::memcpy(&config.bias_bits, (const char *) output->op_params + sizeof(float), sizeof(config.bias_bits));

    std::vector<float> input_values;
    std::vector<float> output_values((size_t) config.elements);
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }

    const size_t bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "scale", &attrs);
        status = scale_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_unary(ggml_backend_dayan_context *          ctx,
                                    const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    const DayanElementwiseConfig config = { (uint64_t) ggml_nelements(output) };
    std::vector<float> input_values;
    std::vector<float> output_values((size_t) config.elements);
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }

    const size_t bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        const char * launch_name = nullptr;
        decltype(&sigmoid_f32_1c) kernel = nullptr;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::sigmoid:
                launch_name = "sigmoid";
                kernel      = sigmoid_f32_1c;
                break;
            case ggml_backend_dayan_slot_kind::silu:
                launch_name = "silu";
                kernel      = silu_f32_1c;
                break;
            case ggml_backend_dayan_slot_kind::softplus:
                launch_name = "softplus";
                kernel      = softplus_f32_1c;
                break;
            case ggml_backend_dayan_slot_kind::unary_math:
                switch (output->op) {
                    case GGML_OP_SQR:
                        launch_name = "square";
                        kernel      = square_f32_1c;
                        break;
                    case GGML_OP_SQRT:
                        launch_name = "sqrt";
                        kernel      = sqrt_f32_1c;
                        break;
                    case GGML_OP_LOG:
                        launch_name = "log";
                        kernel      = log_f32_1c;
                        break;
                    case GGML_OP_SIN:
                        launch_name = "sin";
                        kernel      = sin_f32_1c;
                        break;
                    case GGML_OP_COS:
                        launch_name = "cos";
                        kernel      = cos_f32_1c;
                        break;
                    case GGML_OP_UNARY:
                        switch (ggml_get_unary_op(output)) {
                            case GGML_UNARY_OP_ABS:
                                launch_name = "abs";
                                kernel      = abs_f32_1c;
                                break;
                            case GGML_UNARY_OP_NEG:
                                launch_name = "neg";
                                kernel      = neg_f32_1c;
                                break;
                            case GGML_UNARY_OP_TANH:
                                launch_name = "tanh";
                                kernel      = tanh_f32_1c;
                                break;
                            case GGML_UNARY_OP_RELU:
                                launch_name = "relu";
                                kernel      = relu_f32_1c;
                                break;
                            case GGML_UNARY_OP_GELU:
                                launch_name = "gelu";
                                kernel      = gelu_f32_1c;
                                break;
                            case GGML_UNARY_OP_GELU_QUICK:
                                launch_name = "gelu-quick";
                                kernel      = gelu_quick_f32_1c;
                                break;
                            case GGML_UNARY_OP_EXP:
                                launch_name = "exp";
                                kernel      = exp_f32_1c;
                                break;
                            default:
                                break;
                        }
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
        if (kernel == nullptr) {
            status = -1;
        } else {
            ggml_backend_dayan_init_launch_attrs(ctx, launch_name, &attrs);
            status = kernel(buffers.config, buffers.weight0, buffers.output, &attrs);
        }
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_reduce(ggml_backend_dayan_context *          ctx,
                                     const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanReduceConfig   config = {
        slot.kind == ggml_backend_dayan_slot_kind::sum ? (uint32_t) ggml_nelements(input) : (uint32_t) input->ne[0],
        slot.kind == ggml_backend_dayan_slot_kind::sum ? 1U : (uint32_t) ggml_nrows(input),
        0,
        0,
    };
    const float scale = slot.kind == ggml_backend_dayan_slot_kind::mean ? 1.0f / (float) config.width : 1.0f;
    std::memcpy(&config.scale_bits, &scale, sizeof(scale));

    std::vector<float> input_values;
    std::vector<float> output_values((size_t) ggml_nelements(output));
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }

    const size_t input_bytes  = input_values.size() * sizeof(float);
    const size_t output_bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, input_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), input_bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::sum:
                ggml_backend_dayan_init_launch_attrs(ctx, "sum", &attrs);
                status = sum_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
                break;
            case ggml_backend_dayan_slot_kind::sum_rows:
                ggml_backend_dayan_init_launch_attrs(ctx, "sum-rows", &attrs);
                status = sum_rows_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
                break;
            case ggml_backend_dayan_slot_kind::mean:
                ggml_backend_dayan_init_launch_attrs(ctx, "mean", &attrs);
                status = mean_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
                break;
            default:
                status = -1;
                break;
        }
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_l2_norm(ggml_backend_dayan_context *          ctx,
                                      const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanNormConfig     config = ggml_backend_dayan_norm_config(output);
    std::vector<float> input_values;
    std::vector<float> output_values((size_t) ggml_nelements(output));
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }

    const size_t bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "l2-norm", &attrs);
        status = l2_norm_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_get_rows(ggml_backend_dayan_context *          ctx,
                                       const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output  = slot.nodes[0];
    const ggml_tensor * source  = output->src[0];
    const ggml_tensor * indices = output->src[1];
    if (ggml_nelements(output) == 0) {
        return 0;
    }

    uint32_t            source_type;
    if (!ggml_backend_dayan_type_code(source->type, &source_type)) {
        return -1;
    }

    const uint32_t rows = (uint32_t) ggml_nrows(output);
    const size_t row_bytes = (size_t) output->ne[0] * ggml_type_size(source->type);
    std::vector<uint8_t> packed_source((size_t) rows * row_bytes);
    std::vector<uint64_t> flattened_indices(rows);
    size_t target = 0;
    for (int64_t index2 = 0; index2 < indices->ne[2]; ++index2) {
        for (int64_t index1 = 0; index1 < indices->ne[1]; ++index1) {
            for (int64_t index0 = 0; index0 < indices->ne[0]; ++index0) {
                uint64_t source_row;
                if (!ggml_backend_dayan_read_index(indices, index0, index1, index2, &source_row) ||
                    source_row >= (uint64_t) source->ne[1]) {
                    return -1;
                }
                const char * source_data = (const char *) source->data + index2 * source->nb[3] +
                                           index1 * source->nb[2] + source_row * source->nb[1];
                uint8_t * destination = packed_source.data() + target * row_bytes;
                if (source->nb[0] == ggml_type_size(source->type)) {
                    std::memcpy(destination, source_data, row_bytes);
                } else {
                    for (int64_t column = 0; column < output->ne[0]; ++column) {
                        std::memcpy(destination + (size_t) column * ggml_type_size(source->type),
                                    source_data + column * source->nb[0], ggml_type_size(source->type));
                    }
                }
                flattened_indices[target] = target;
                ++target;
            }
        }
    }
    if (target != flattened_indices.size()) {
        return -1;
    }

    const DayanRowsConfig config = {
        (uint32_t) output->ne[0],
        rows,
        rows,
        rows,
        source_type,
        kDayanDataTypeF32,
    };
    std::vector<float> output_values((size_t) ggml_nelements(output));
    const size_t source_bytes = packed_source.size();
    const size_t output_bytes = output_values.size() * sizeof(float);
    const size_t index_bytes  = flattened_indices.size() * sizeof(uint64_t);

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, source_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, index_bytes, &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, packed_source.data(), source_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, flattened_indices.data(), index_bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "get-rows", &attrs);
        status = get_rows_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_set_rows(ggml_backend_dayan_context *          ctx,
                                       const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output  = slot.nodes[0];
    const ggml_tensor * source  = output->src[0];
    const ggml_tensor * indices = output->src[1];
    uint32_t            destination_type;
    if (!ggml_backend_dayan_type_code(output->type, &destination_type)) {
        return -1;
    }

    std::vector<float>   source_values;
    std::vector<uint8_t> destination_values;
    if (!ggml_backend_dayan_pack_f32(source, &source_values) ||
        !ggml_backend_dayan_pack_scalar(output, &destination_values)) {
        return -1;
    }

    const uint32_t rows = (uint32_t) ggml_nrows(source);
    std::vector<uint64_t> flattened_indices(rows);
    size_t target = 0;
    for (int64_t index3 = 0; index3 < source->ne[3]; ++index3) {
        for (int64_t index2 = 0; index2 < source->ne[2]; ++index2) {
            for (int64_t index1 = 0; index1 < source->ne[1]; ++index1) {
                uint64_t destination_row;
                if (!ggml_backend_dayan_read_index(indices, index1, index2 % indices->ne[1],
                                                   index3 % indices->ne[2], &destination_row) ||
                    destination_row >= (uint64_t) output->ne[1]) {
                    return -1;
                }
                flattened_indices[target++] =
                    ((uint64_t) index3 * (uint64_t) output->ne[2] + (uint64_t) index2) *
                        (uint64_t) output->ne[1] +
                    destination_row;
            }
        }
    }
    if (target != flattened_indices.size()) {
        return -1;
    }

    const DayanRowsConfig config = {
        (uint32_t) source->ne[0],
        rows,
        rows,
        (uint64_t) ggml_nrows(output),
        kDayanDataTypeF32,
        destination_type,
    };
    const size_t source_bytes = source_values.size() * sizeof(float);
    const size_t index_bytes  = flattened_indices.size() * sizeof(uint64_t);

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, source_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, index_bytes, &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, destination_values.size(), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, source_values.data(), source_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, flattened_indices.data(), index_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.output, destination_values.data(),
                                           destination_values.size());
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "set-rows", &attrs);
        status = set_rows_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, destination_values.data(), buffers.output,
                                             destination_values.size());
    }
    if (status == 0 && !ggml_backend_dayan_unpack_scalar(output, destination_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_copy_tensor(ggml_backend_dayan_context *          ctx,
                                          const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * source = output->src[0];
    if (ggml_nelements(output) == 0) {
        return 0;
    }

    uint32_t source_type;
    uint32_t destination_type;
    if (!ggml_backend_dayan_type_code(source->type, &source_type) ||
        !ggml_backend_dayan_type_code(output->type, &destination_type)) {
        return -1;
    }

    std::vector<uint8_t> source_values;
    std::vector<uint8_t> output_values((size_t) ggml_nelements(output) * ggml_type_size(output->type));
    if (!ggml_backend_dayan_pack_scalar(source, &source_values)) {
        return -1;
    }

    const DayanCopyConfig config = {
        (uint64_t) ggml_nelements(output),
        source_type,
        destination_type,
    };
    const size_t source_bytes = source_values.size();

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, source_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_values.size(), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, source_values.data(), source_bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "copy", &attrs);
        status = copy_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_values.size());
    }
    if (status == 0 && !ggml_backend_dayan_unpack_scalar(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_repeat(ggml_backend_dayan_context *          ctx,
                                     const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * source = output->src[0];
    if (ggml_nelements(output) == 0) {
        return 0;
    }

    DayanRepeatConfig config = {};
    if (!ggml_backend_dayan_type_code(output->type, &config.type)) {
        return -1;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.source_ne[dimension] = (uint32_t) source->ne[dimension];
        config.output_ne[dimension] = (uint32_t) output->ne[dimension];
    }

    std::vector<uint8_t> source_values;
    std::vector<uint8_t> output_values((size_t) ggml_nelements(output) * ggml_type_size(output->type));
    if (!ggml_backend_dayan_pack_scalar(source, &source_values)) {
        return -1;
    }

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, source_values.size(), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_values.size(), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, source_values.data(), source_values.size());
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "repeat", &attrs);
        status = repeat_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_values.size());
    }
    if (status == 0 && !ggml_backend_dayan_unpack_scalar(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_set(ggml_backend_dayan_context *          ctx,
                                  const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * base   = output->src[0];
    const ggml_tensor * source = output->src[1];
    std::vector<uint8_t> base_values;
    std::vector<uint8_t> source_values;
    std::vector<uint8_t> output_values(ggml_nbytes(output));
    if (!ggml_backend_dayan_pack_scalar(base, &base_values) ||
        !ggml_backend_dayan_pack_scalar(source, &source_values)) {
        return -1;
    }

    DayanSetConfig config = {
        (uint64_t) output_values.size(),
        { 0, 0, 0, 0 },
        (uint32_t) ggml_type_size(output->type),
        { (uint32_t) output->op_params[0], (uint32_t) output->op_params[1], (uint32_t) output->op_params[2] },
        (uint32_t) output->op_params[3],
    };
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.source_ne[dimension] = (uint32_t) source->ne[dimension];
    }

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, base_values.size(), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, source_values.size(), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_values.size(), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, base_values.data(), base_values.size());
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, source_values.data(), source_values.size());
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "set", &attrs);
        status = set_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_values.size());
    }
    if (status == 0 && !ggml_backend_dayan_unpack_scalar(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_layout(ggml_backend_dayan_context *          ctx,
                                     const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanLayoutConfig   config = {};
    if (!ggml_backend_dayan_type_code(output->type, &config.type)) {
        return -1;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.input_ne[dimension]  = (uint32_t) input->ne[dimension];
        config.output_ne[dimension] = (uint32_t) output->ne[dimension];
    }
    for (int parameter = 0; parameter < 8; ++parameter) {
        config.parameters[parameter] = output->op_params[parameter];
    }

    std::vector<float> input_values;
    std::vector<float> output_values((size_t) ggml_nelements(output));
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }

    const size_t input_bytes  = input_values.size() * sizeof(float);
    const size_t output_bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, input_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), input_bytes);
    }
    if (status == 0) {
        using kernel_t = int (*)(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, const npu_kernel_launch_attrs_t *);
        kernel_t    kernel      = nullptr;
        const char * launch_name = nullptr;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::diag:
                kernel      = diag_f32_1c;
                launch_name = "diag";
                break;
            case ggml_backend_dayan_slot_kind::diag_mask_inf:
                kernel      = diag_mask_inf_f32_1c;
                launch_name = "diag-mask-inf";
                break;
            case ggml_backend_dayan_slot_kind::diag_mask_zero:
                kernel      = diag_mask_zero_f32_1c;
                launch_name = "diag-mask-zero";
                break;
            case ggml_backend_dayan_slot_kind::pad:
                kernel      = pad_f32_1c;
                launch_name = "pad";
                break;
            case ggml_backend_dayan_slot_kind::pad_circular:
                kernel      = pad_circular_f32_1c;
                launch_name = "pad-circular";
                break;
            case ggml_backend_dayan_slot_kind::pad_reflect_1d:
                kernel      = pad_reflect_1d_f32_1c;
                launch_name = "pad-reflect-1d";
                break;
            case ggml_backend_dayan_slot_kind::roll:
                kernel      = roll_f32_1c;
                launch_name = "roll";
                break;
            default:
                break;
        }
        if (kernel == nullptr) {
            status = -1;
        } else {
            npu_kernel_launch_attrs_t attrs;
            ggml_backend_dayan_init_launch_attrs(ctx, launch_name, &attrs);
            status = kernel(buffers.config, buffers.weight0, buffers.output, &attrs);
        }
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_clamp(ggml_backend_dayan_context *          ctx,
                                    const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanClampConfig    config = { (uint64_t) ggml_nelements(output), 0, 0 };
    std::memcpy(&config.min_bits, output->op_params, sizeof(config.min_bits));
    std::memcpy(&config.max_bits, (const char *) output->op_params + sizeof(float), sizeof(config.max_bits));

    std::vector<float> input_values;
    std::vector<float> output_values((size_t) config.elements);
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }

    const size_t bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "clamp", &attrs);
        status = clamp_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_softmax(ggml_backend_dayan_context *          ctx,
                                      const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    const ggml_tensor * mask   = output->src[1];
    const ggml_tensor * sinks  = output->src[2];

    DayanSoftmaxConfig config = {
        (uint32_t) output->ne[0],
        (uint32_t) output->ne[1],
        (uint32_t) output->ne[2],
        (uint32_t) output->ne[3],
        mask != nullptr ? (uint32_t) mask->ne[1] : 1U,
        mask != nullptr ? (uint32_t) mask->ne[2] : 1U,
        mask != nullptr ? (uint32_t) mask->ne[3] : 1U,
        kDayanDataTypeF32,
        0,
        0,
        (mask != nullptr ? kDayanSoftmaxHasMask : 0U) | (sinks != nullptr ? kDayanSoftmaxHasSinks : 0U),
        0,
    };
    std::memcpy(&config.scale_bits, output->op_params, sizeof(config.scale_bits));
    std::memcpy(&config.max_bias_bits,
                (const char *) output->op_params + sizeof(float),
                sizeof(config.max_bias_bits));
    if (mask != nullptr && !ggml_backend_dayan_type_code(mask->type, &config.mask_type)) {
        return -1;
    }

    std::vector<float> input_values;
    std::vector<uint8_t> mask_values(sizeof(float), 0);
    std::vector<float> sink_values(1, 0.0f);
    std::vector<float> output_values((size_t) ggml_nelements(output));
    if (!ggml_backend_dayan_pack_f32(input, &input_values) ||
        (mask != nullptr && !ggml_backend_dayan_pack_scalar(mask, &mask_values)) ||
        (sinks != nullptr && !ggml_backend_dayan_pack_f32(sinks, &sink_values))) {
        return -1;
    }

    const size_t input_bytes  = input_values.size() * sizeof(float);
    const size_t output_bytes = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, input_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, mask_values.size(), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sink_values.size() * sizeof(float), &buffers.input2);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), input_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, mask_values.data(), mask_values.size());
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input2, sink_values.data(),
                                           sink_values.size() * sizeof(float));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "softmax", &attrs);
        status = softmax_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2, buffers.output,
                                &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_concat(ggml_backend_dayan_context *          ctx,
                                     const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * left   = output->src[0];
    const ggml_tensor * right  = output->src[1];
    DayanConcatConfig config = {};
    config.axis = (uint32_t) ggml_get_op_params_i32(output, 0);
    if (!ggml_backend_dayan_type_code(output->type, &config.type)) {
        return -1;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.left_ne[dimension]   = (uint32_t) left->ne[dimension];
        config.right_ne[dimension]  = (uint32_t) right->ne[dimension];
        config.output_ne[dimension] = (uint32_t) output->ne[dimension];
    }

    std::vector<uint8_t> left_values;
    std::vector<uint8_t> right_values;
    std::vector<uint8_t> output_values((size_t) ggml_nelements(output) * ggml_type_size(output->type));
    if (!ggml_backend_dayan_pack_scalar(left, &left_values) ||
        !ggml_backend_dayan_pack_scalar(right, &right_values)) {
        return -1;
    }

    const size_t left_bytes   = left_values.size();
    const size_t right_bytes  = right_values.size();
    const size_t output_bytes = output_values.size();

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, left_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, right_bytes, &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, left_values.data(), left_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, right_values.data(), right_bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "concat", &attrs);
        status = concat_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_scalar(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_rope(ggml_backend_dayan_context *          ctx,
                                   const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output    = slot.nodes[0];
    const ggml_tensor * input     = output->src[0];
    const ggml_tensor * positions = output->src[1];
    const ggml_tensor * factors   = output->src[2];
    const int32_t       n_dims    = ggml_get_op_params_i32(output, 1);
    const int32_t       mode      = ggml_get_op_params_i32(output, 2);

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    std::memcpy(&freq_base, (const int32_t *) output->op_params + 5, sizeof(freq_base));
    std::memcpy(&freq_scale, (const int32_t *) output->op_params + 6, sizeof(freq_scale));
    std::memcpy(&ext_factor, (const int32_t *) output->op_params + 7, sizeof(ext_factor));
    std::memcpy(&attn_factor, (const int32_t *) output->op_params + 8, sizeof(attn_factor));
    std::memcpy(&beta_fast, (const int32_t *) output->op_params + 9, sizeof(beta_fast));
    std::memcpy(&beta_slow, (const int32_t *) output->op_params + 10, sizeof(beta_slow));

    const float theta_scale = std::pow(freq_base, -2.0f / (float) n_dims);
    float       mscale      = attn_factor;
    if (ext_factor != 0.0f) {
        mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    }
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, ggml_get_op_params_i32(output, 4), freq_base, beta_fast, beta_slow, corr_dims);

    DayanRopeConfig config = {
        (uint32_t) output->ne[0],
        (uint32_t) output->ne[1],
        (uint32_t) output->ne[2],
        (uint32_t) output->ne[3],
        (uint32_t) n_dims,
        (uint32_t) mode,
        0,
        0,
        0,
        0,
        0,
        0,
        { 0, 0, 0, 0 },
    };
    std::memcpy(&config.theta_scale_bits, &theta_scale, sizeof(theta_scale));
    std::memcpy(&config.freq_scale_bits, &freq_scale, sizeof(freq_scale));
    std::memcpy(&config.ext_factor_bits, &ext_factor, sizeof(ext_factor));
    std::memcpy(&config.mscale_bits, &mscale, sizeof(mscale));
    std::memcpy(&config.corr_dim0_bits, &corr_dims[0], sizeof(corr_dims[0]));
    std::memcpy(&config.corr_dim1_bits, &corr_dims[1], sizeof(corr_dims[1]));
    std::memcpy(config.sections, (const int32_t *) output->op_params + 11, sizeof(config.sections));

    std::vector<float> input_values;
    std::vector<float> output_values((size_t) ggml_nelements(output));
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }

    const uint32_t position_axes = (mode & GGML_ROPE_TYPE_MROPE) != 0 ? 4U : 1U;
    std::vector<int32_t> position_values((size_t) position_axes * config.tokens);
    for (size_t index = 0; index < position_values.size(); ++index) {
        const char * address = (const char *) positions->data + index * positions->nb[0];
        std::memcpy(&position_values[index], address, sizeof(position_values[index]));
    }

    std::vector<float> factor_values((size_t) n_dims / 2, 1.0f);
    if (factors != nullptr) {
        std::vector<float> packed_factors;
        if (!ggml_backend_dayan_pack_f32(factors, &packed_factors) || packed_factors.size() < factor_values.size()) {
            return -1;
        }
        std::copy_n(packed_factors.begin(), factor_values.size(), factor_values.begin());
    }

    const size_t input_bytes    = input_values.size() * sizeof(float);
    const size_t position_bytes = position_values.size() * sizeof(int32_t);
    const size_t factor_bytes   = factor_values.size() * sizeof(float);
    const size_t output_bytes   = output_values.size() * sizeof(float);
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, input_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, position_bytes, &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, factor_bytes, &buffers.input2);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, output_bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), input_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, position_values.data(), position_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input2, factor_values.data(), factor_bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "rope", &attrs);
        status = rope_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, output_bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_residual_add(ggml_backend_dayan_context *          ctx,
                                           const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * add = slot.nodes[0];
    const DayanElementwiseConfig config = {
        (uint64_t) ggml_nelements(add),
    };
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(add->src[0]), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(add->src[1]), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(add), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, add->src[0]->data, ggml_nbytes(add->src[0]));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, add->src[1]->data, ggml_nbytes(add->src[1]));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "residual-add", &attrs);
        status = residual_add_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, add->data, buffers.output, ggml_nbytes(add));
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan residual add failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_norm_raw(ggml_backend_dayan_context *          ctx,
                                       const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    const bool    rms    = slot.kind == ggml_backend_dayan_slot_kind::rms_norm;
    const DayanNormConfig config = ggml_backend_dayan_norm_config(output);
    std::vector<float> input_values;
    std::vector<float> output_values((size_t) ggml_nelements(output));
    if (!ggml_backend_dayan_pack_f32(output->src[0], &input_values)) {
        return -1;
    }
    const size_t bytes = output_values.size() * sizeof(float);

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, rms ? "rms-norm" : "norm", &attrs);
        status = rms ? rms_norm_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs) :
                       norm_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan %s failed with status %d\n", __func__, rms ? "RMSNorm" : "Norm", status);
    }
    return status;
}

static int ggml_backend_dayan_group_norm(ggml_backend_dayan_context *          ctx,
                                         const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanGroupNormConfig config = {
        (uint32_t) output->ne[0],
        (uint32_t) output->ne[1],
        (uint32_t) output->ne[2],
        (uint32_t) output->ne[3],
        (uint32_t) ggml_get_op_params_i32(output, 0),
        0,
    };
    const float epsilon = ggml_get_op_params_f32(output, 1);
    std::memcpy(&config.epsilon_bits, &epsilon, sizeof(epsilon));

    std::vector<float> input_values;
    std::vector<float> output_values((size_t) ggml_nelements(output));
    if (!ggml_backend_dayan_pack_f32(input, &input_values)) {
        return -1;
    }
    const size_t bytes = input_values.size() * sizeof(float);

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input_values.data(), bytes);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "group-norm", &attrs);
        status = group_norm_f32_1c(buffers.config, buffers.weight0, buffers.output, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, output_values.data(), buffers.output, bytes);
    }
    if (status == 0 && !ggml_backend_dayan_unpack_f32(output, output_values)) {
        status = -1;
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_rms_norm_scaled(ggml_backend_dayan_context *          ctx,
                                              ggml_backend_dayan_graph_buffers *     graph_buffers,
                                              const ggml_backend_dayan_graph_slot & slot) {
    const bool has_add = slot.kind == ggml_backend_dayan_slot_kind::add_rms_norm_scaled ||
                         slot.kind == ggml_backend_dayan_slot_kind::add_rms_norm_quantized ||
                         slot.kind == ggml_backend_dayan_slot_kind::add_rms_norm_mul_mat;
    const bool has_lowp = slot.kind == ggml_backend_dayan_slot_kind::add_rms_norm_quantized ||
                          slot.kind == ggml_backend_dayan_slot_kind::add_rms_norm_mul_mat ||
                          slot.kind == ggml_backend_dayan_slot_kind::rms_norm_quantized ||
                          slot.kind == ggml_backend_dayan_slot_kind::rms_norm_mul_mat;
    ggml_tensor * add    = has_add ? slot.nodes[0] : nullptr;
    ggml_tensor * rms    = has_add ? slot.nodes[1] : slot.nodes[0];
    ggml_tensor * scaled = has_add ? slot.nodes[2] : slot.nodes[1];
    ggml_tensor * lowp   = has_lowp ? slot.nodes[has_add ? 3 : 2] : nullptr;
    const ggml_tensor * input = has_add ? add->src[0] : rms->src[0];
    const ggml_tensor * right = has_add ? add->src[1] : nullptr;
    const ggml_tensor * scale = ggml_backend_dayan_norm_scale(scaled, rms);
    const DayanNormConfig config = ggml_backend_dayan_norm_config(rms);

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(input), &buffers.weight0);
    }
    if (status == 0 && has_add) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(right), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(scale), &buffers.input2);
    }
    if (status == 0 && has_add) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(add), &buffers.input3);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(rms), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(scaled), &buffers.output1);
    }
    if (status == 0 && lowp != nullptr) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(lowp), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input->data, ggml_nbytes(input));
    }
    if (status == 0 && has_add) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, right->data, ggml_nbytes(right));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input2, scale->data, ggml_nbytes(scale));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, has_add ? "add-rms" : "rms-scale", &attrs);
        if (has_add) {
            if (lowp == nullptr) {
                status = add_rms_norm_scale_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                                   buffers.input3, buffers.output, buffers.output1, &attrs);
            } else if (lowp->type == GGML_TYPE_BF16) {
                status = add_rms_norm_scale_bf16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                                    buffers.input3, buffers.output, buffers.output1, buffers.perf,
                                                    &attrs);
            } else {
                status = add_rms_norm_scale_fp16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                                    buffers.input3, buffers.output, buffers.output1, buffers.perf,
                                                    &attrs);
            }
        } else if (lowp == nullptr) {
            status = rms_norm_scale_f32_1c(buffers.config, buffers.weight0, buffers.input2, buffers.output,
                                           buffers.output1, &attrs);
        } else if (lowp->type == GGML_TYPE_BF16) {
            status = rms_norm_scale_bf16_1c(buffers.config, buffers.weight0, buffers.input2, buffers.output,
                                            buffers.output1, buffers.perf, &attrs);
        } else {
            status = rms_norm_scale_fp16_1c(buffers.config, buffers.weight0, buffers.input2, buffers.output,
                                            buffers.output1, buffers.perf, &attrs);
        }
    }
    if (status == 0 && has_add) {
        status = ggml_backend_dayan_download(ctx->stream, add->data, buffers.input3, ggml_nbytes(add));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, rms->data, buffers.output, ggml_nbytes(rms));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, scaled->data, buffers.output1, ggml_nbytes(scaled));
    }
    if (status == 0 && lowp != nullptr) {
        status = ggml_backend_dayan_download(ctx->stream, lowp->data, buffers.perf, ggml_nbytes(lowp));
    }
    if (status == 0 && lowp != nullptr) {
        status = ggml_backend_dayan_register_activation(graph_buffers, lowp, lowp->type, &buffers.perf);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan fused RMSNorm failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_rms_norm_matmul(ggml_backend_dayan_context *          ctx,
                                              ggml_backend_dayan_graph_buffers *     graph_buffers,
                                              const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * rms    = slot.nodes[0];
    ggml_tensor * scaled = slot.nodes[1];
    ggml_tensor * lowp   = slot.nodes[2];
    ggml_tensor * matmul = slot.nodes[3];
    const ggml_tensor * input  = rms->src[0];
    const ggml_tensor * scale  = ggml_backend_dayan_norm_scale(scaled, rms);
    const ggml_tensor * weight = matmul->src[0];

    if (ggml_nbytes(weight) > GGML_DAYAN_MATMUL_BUFFER_BYTES ||
        ggml_nbytes(matmul) > GGML_DAYAN_MATMUL_BUFFER_BYTES) {
        int status = ggml_backend_dayan_rms_norm_scaled(ctx, graph_buffers, slot);
        npu_buffer_t * activation = nullptr;
        if (status == 0) {
            status = ggml_backend_dayan_prepare_activation(ctx, graph_buffers, lowp, weight->type, &activation);
        }
        if (status == 0) {
            status = ggml_backend_dayan_mul_mat(ctx, matmul, activation);
        }
        return status;
    }

    const DayanNormMatmulConfig config = {
        ggml_backend_dayan_norm_config(rms),
        {
            (uint64_t) lowp->ne[1],
            (uint64_t) weight->ne[1],
            (uint64_t) weight->ne[0],
            lowp->nb[1],
            weight->nb[1],
            matmul->nb[1],
        },
    };
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(input), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(scale), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(rms), &buffers.input2);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(scaled), &buffers.input3);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(lowp), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(weight), &buffers.output1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(matmul), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input->data, ggml_nbytes(input));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, scale->data, ggml_nbytes(scale));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.output1, weight->data, ggml_nbytes(weight));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "rms-matmul", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     rms_norm_matmul_bf16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                             buffers.input3, buffers.output, buffers.output1, buffers.perf, &attrs) :
                     rms_norm_matmul_fp16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                             buffers.input3, buffers.output, buffers.output1, buffers.perf, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, rms->data, buffers.input2, ggml_nbytes(rms));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, scaled->data, buffers.input3, ggml_nbytes(scaled));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, lowp->data, buffers.output, ggml_nbytes(lowp));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, matmul->data, buffers.perf, ggml_nbytes(matmul));
    }
    if (status == 0) {
        status = ggml_backend_dayan_register_activation(graph_buffers, lowp, lowp->type, &buffers.output);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan fused RMSNorm matrix multiplication failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_add_rms_norm_matmul(ggml_backend_dayan_context *          ctx,
                                                  ggml_backend_dayan_graph_buffers *     graph_buffers,
                                                  const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * add    = slot.nodes[0];
    ggml_tensor * rms    = slot.nodes[1];
    ggml_tensor * scaled = slot.nodes[2];
    ggml_tensor * lowp   = slot.nodes[3];
    ggml_tensor * matmul = slot.nodes[4];
    const ggml_tensor * left   = add->src[0];
    const ggml_tensor * right  = add->src[1];
    const ggml_tensor * scale  = ggml_backend_dayan_norm_scale(scaled, rms);
    const ggml_tensor * weight = matmul->src[0];

    if (ggml_nbytes(weight) > GGML_DAYAN_MATMUL_BUFFER_BYTES ||
        ggml_nbytes(matmul) > GGML_DAYAN_MATMUL_BUFFER_BYTES) {
        int status = ggml_backend_dayan_rms_norm_scaled(ctx, graph_buffers, slot);
        npu_buffer_t * activation = nullptr;
        if (status == 0) {
            status = ggml_backend_dayan_prepare_activation(ctx, graph_buffers, lowp, weight->type, &activation);
        }
        if (status == 0) {
            status = ggml_backend_dayan_mul_mat(ctx, matmul, activation);
        }
        return status;
    }

    const DayanNormMatmulConfig config = {
        ggml_backend_dayan_norm_config(rms),
        {
            (uint64_t) lowp->ne[1],
            (uint64_t) weight->ne[1],
            (uint64_t) weight->ne[0],
            lowp->nb[1],
            weight->nb[1],
            matmul->nb[1],
        },
    };
    const size_t f32_bytes  = ggml_nbytes(rms);
    const size_t lowp_bytes = ggml_nbytes(lowp);
    if (f32_bytes > (SIZE_MAX - lowp_bytes) / 3) {
        return -1;
    }
    const size_t rms_offset    = f32_bytes;
    const size_t scaled_offset = 2 * f32_bytes;
    const size_t lowp_offset   = 3 * f32_bytes;
    const size_t bundle_bytes  = lowp_offset + lowp_bytes;

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(left), &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(right), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(scale), &buffers.input2);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, bundle_bytes, &buffers.input3);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(weight), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(matmul), &buffers.output1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, lowp_bytes, &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, left->data, ggml_nbytes(left));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, right->data, ggml_nbytes(right));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input2, scale->data, ggml_nbytes(scale));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.output, weight->data, ggml_nbytes(weight));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "add-rms-matmul", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     add_rms_norm_matmul_bf16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                                 buffers.input3, buffers.output, buffers.output1, &attrs) :
                     add_rms_norm_matmul_fp16_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.input2,
                                                 buffers.input3, buffers.output, buffers.output1, &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download_at(ctx->stream, add->data, buffers.input3, 0, ggml_nbytes(add));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download_at(ctx->stream, rms->data, buffers.input3, rms_offset, ggml_nbytes(rms));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download_at(ctx->stream, scaled->data, buffers.input3, scaled_offset,
                                                ggml_nbytes(scaled));
    }
    if (status == 0) {
        status = ggml_backend_dayan_download_at(ctx->stream, lowp->data, buffers.input3, lowp_offset, lowp_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, matmul->data, buffers.output1, ggml_nbytes(matmul));
    }
    if (status == 0) {
        status = ggml_backend_dayan_copy(ctx->stream, buffers.perf, 0, buffers.input3, lowp_offset, lowp_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_register_activation(graph_buffers, lowp, lowp->type, &buffers.perf);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan fused residual RMSNorm matrix multiplication failed with status %d\n", __func__,
                       status);
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

static int ggml_backend_dayan_gated_delta_net(ggml_backend_dayan_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    DayanGatedDeltaNetConfig config = {};
    config.head_dim                 = (uint32_t) v->ne[0];
    config.heads                    = (uint32_t) v->ne[1];
    config.tokens                   = (uint32_t) v->ne[2];
    config.sequences                = (uint32_t) v->ne[3];
    config.snapshots                = (uint32_t) ggml_get_op_params_i32(dst, 0);
    config.gate_width               = (uint32_t) g->ne[0];
    const float scale               = 1.0f / std::sqrt((float) config.head_dim);
    std::memcpy(&config.scale_bits, &scale, sizeof(scale));

    std::vector<float> input;
    config.q_offset = input.size() * sizeof(float);
    if (!ggml_backend_dayan_append_gdn_tensor(q, config.heads, config.tokens, config.sequences, &input)) {
        return -1;
    }
    config.k_offset = input.size() * sizeof(float);
    if (!ggml_backend_dayan_append_gdn_tensor(k, config.heads, config.tokens, config.sequences, &input)) {
        return -1;
    }
    config.v_offset = input.size() * sizeof(float);
    if (!ggml_backend_dayan_append_gdn_tensor(v, config.heads, config.tokens, config.sequences, &input)) {
        return -1;
    }
    config.g_offset = input.size() * sizeof(float);
    ggml_backend_dayan_append_contiguous_f32(g, &input);
    config.beta_offset = input.size() * sizeof(float);
    ggml_backend_dayan_append_contiguous_f32(beta, &input);
    config.input_bytes = input.size() * sizeof(float);

    ggml_backend_dayan_buffers buffers;
    int64_t                    device_status = -22;
    int                        status        = 0;
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, config.input_bytes, &buffers.weight0);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(state), &buffers.weight1);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, ggml_nbytes(dst), &buffers.output);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(device_status), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight0, input.data(), config.input_bytes);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.weight1, state->data, ggml_nbytes(state));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-gdn", &attrs);
        status = gated_delta_net_f32_1c(buffers.config, buffers.weight0, buffers.weight1, buffers.output, buffers.perf,
                                        &attrs);
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, &device_status, buffers.perf, sizeof(device_status));
    }
    if (status == 0 && device_status < 0) {
        status = (int) device_status;
    }
    if (status == 0) {
        status = ggml_backend_dayan_download(ctx->stream, dst->data, buffers.output, ggml_nbytes(dst));
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan Gated Delta Net failed with status %d\n", __func__, status);
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
            case ggml_backend_dayan_slot_kind::elementwise_add:
            case ggml_backend_dayan_slot_kind::elementwise_mul:
            case ggml_backend_dayan_slot_kind::elementwise_sub:
            case ggml_backend_dayan_slot_kind::elementwise_div:
                status = ggml_backend_dayan_elementwise_binary(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::scale:
                status = ggml_backend_dayan_scale(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::sigmoid:
            case ggml_backend_dayan_slot_kind::silu:
            case ggml_backend_dayan_slot_kind::softplus:
            case ggml_backend_dayan_slot_kind::unary_math:
                status = ggml_backend_dayan_unary(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::sum:
            case ggml_backend_dayan_slot_kind::sum_rows:
            case ggml_backend_dayan_slot_kind::mean:
                status = ggml_backend_dayan_reduce(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::l2_norm:
                status = ggml_backend_dayan_l2_norm(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::get_rows:
                status = ggml_backend_dayan_get_rows(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::set_rows:
                status = ggml_backend_dayan_set_rows(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::copy:
                status = ggml_backend_dayan_copy_tensor(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::repeat:
                status = ggml_backend_dayan_repeat(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::set:
                status = ggml_backend_dayan_set(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::diag:
            case ggml_backend_dayan_slot_kind::diag_mask_inf:
            case ggml_backend_dayan_slot_kind::diag_mask_zero:
            case ggml_backend_dayan_slot_kind::pad:
            case ggml_backend_dayan_slot_kind::pad_circular:
            case ggml_backend_dayan_slot_kind::pad_reflect_1d:
            case ggml_backend_dayan_slot_kind::roll:
                status = ggml_backend_dayan_layout(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::clamp:
                status = ggml_backend_dayan_clamp(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::softmax:
                status = ggml_backend_dayan_softmax(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::concat:
                status = ggml_backend_dayan_concat(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::rope:
                status = ggml_backend_dayan_rope(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::residual_add:
                status = ggml_backend_dayan_residual_add(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::rms_norm:
            case ggml_backend_dayan_slot_kind::norm:
                status = ggml_backend_dayan_norm_raw(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::group_norm:
                status = ggml_backend_dayan_group_norm(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::rms_norm_scaled:
            case ggml_backend_dayan_slot_kind::rms_norm_quantized:
            case ggml_backend_dayan_slot_kind::add_rms_norm_scaled:
            case ggml_backend_dayan_slot_kind::add_rms_norm_quantized:
                status = ggml_backend_dayan_rms_norm_scaled(ctx, &graph_buffers, slot);
                break;
            case ggml_backend_dayan_slot_kind::rms_norm_mul_mat:
                status = ggml_backend_dayan_rms_norm_matmul(ctx, &graph_buffers, slot);
                break;
            case ggml_backend_dayan_slot_kind::add_rms_norm_mul_mat:
                status = ggml_backend_dayan_add_rms_norm_matmul(ctx, &graph_buffers, slot);
                break;
            case ggml_backend_dayan_slot_kind::flash_attention:
                status = ggml_backend_dayan_flash_attention(ctx, slot.nodes[0]);
                break;
            case ggml_backend_dayan_slot_kind::gated_delta_net:
                status = ggml_backend_dayan_gated_delta_net(ctx, slot.nodes[0]);
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
            "%s: lowered %d nodes to %zu slots, paired=%zu, fused=%zu, residual=%zu, rms=%zu, rms-matmul=%zu, "
            "attention=%zu, delta=%zu, gated=%zu, conv=%zu, activations=%zu\n",
            __func__, cgraph->n_nodes, plan.slots.size(), plan.paired_mul_mat_slots, plan.fused_mul_mat_slots,
            plan.residual_add_slots, plan.rms_norm_slots, plan.rms_norm_matmul_slots, plan.flash_attention_slots,
            plan.gated_delta_net_slots, plan.gated_output_slots, plan.ssm_conv_slots,
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
            return ggml_backend_dayan_supports_gate_cont(op) || ggml_backend_dayan_supports_layout_copy(op);
        case GGML_OP_DUP:
            return ggml_backend_dayan_supports_layout_copy(op);
        case GGML_OP_REPEAT:
            return ggml_backend_dayan_supports_repeat(op);
        case GGML_OP_SET:
            return ggml_backend_dayan_supports_set(op);
        case GGML_OP_DIAG:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
            return ggml_backend_dayan_supports_layout(op);
        case GGML_OP_MUL_MAT:
            return ggml_backend_dayan_supports_mul_mat(op);
        case GGML_OP_GLU:
            return ggml_backend_dayan_supports_glu(op);
        case GGML_OP_CPY:
            return ggml_backend_dayan_supports_quantized_glu(op) || ggml_backend_dayan_supports_quantized_mul_mat(op) ||
                   ggml_backend_dayan_supports_quantized_rms_norm(op) || ggml_backend_dayan_supports_copy(op);
        case GGML_OP_CLAMP:
            return ggml_backend_dayan_supports_clamp(op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_dayan_supports_softmax(op);
        case GGML_OP_GET_ROWS:
            return ggml_backend_dayan_supports_get_rows(op);
        case GGML_OP_SET_ROWS:
            return ggml_backend_dayan_supports_set_rows(op);
        case GGML_OP_CONCAT:
            return ggml_backend_dayan_supports_concat(op);
        case GGML_OP_ROPE:
            return ggml_backend_dayan_supports_rope(op);
        case GGML_OP_ADD:
            return ggml_backend_dayan_supports_residual_add(op) ||
                   ggml_backend_dayan_supports_elementwise_add(op);
        case GGML_OP_SUB:
            return ggml_backend_dayan_supports_elementwise_sub(op);
        case GGML_OP_DIV:
            return ggml_backend_dayan_supports_elementwise_div(op);
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return ggml_backend_dayan_supports_math_unary(op);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_backend_dayan_supports_reduce(op);
        case GGML_OP_NORM:
            return ggml_backend_dayan_supports_norm(op);
        case GGML_OP_RMS_NORM:
            return ggml_backend_dayan_supports_rms_norm(op);
        case GGML_OP_GROUP_NORM:
            return ggml_backend_dayan_supports_group_norm(op);
        case GGML_OP_L2_NORM:
            return ggml_backend_dayan_supports_l2_norm(op);
        case GGML_OP_SCALE:
            return ggml_backend_dayan_supports_scale(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_backend_dayan_supports_flash_attention(op);
        case GGML_OP_GATED_DELTA_NET:
            return ggml_backend_dayan_supports_gated_delta_net(op);
        case GGML_OP_UNARY:
            return ggml_backend_dayan_supports_gate_sigmoid(op) || ggml_backend_dayan_supports_ssm_conv_silu(op) ||
                   ggml_backend_dayan_supports_unary(op) || ggml_backend_dayan_supports_math_unary(op);
        case GGML_OP_MUL:
            return ggml_backend_dayan_supports_gated_attention(op) || ggml_backend_dayan_supports_rms_norm_scale(op) ||
                   ggml_backend_dayan_supports_elementwise_mul(op);
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
