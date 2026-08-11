#include "ggml-dayan.h"

#include "ggml-backend-impl.h"
#include "ggml-dayan-graph.h"
#include "ggml-dayan-kernels.h"
#include "ggml-impl.h"

#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

static constexpr uint32_t GGML_DAYAN_TIMEOUT_MS          = 3600000U;
static constexpr size_t   GGML_DAYAN_MATMUL_BUFFER_BYTES = 16U * 1024U * 1024U;
static constexpr uint64_t GGML_DAYAN_MATMUL_TILE_N       = 384U;

struct ggml_backend_dayan_device_state {
    npu_runtime_t * runtime    = nullptr;
    npu_context_t * context    = nullptr;
    npu_stream_t *  stream     = nullptr;
    size_t          references = 0;
};

struct ggml_backend_dayan_context {
    ggml_backend_dayan_device_state * device  = nullptr;
    npu_runtime_t *                   runtime = nullptr;
    npu_context_t *                   context = nullptr;
    npu_stream_t *                    stream  = nullptr;
};

struct ggml_backend_dayan_buffer_context {
    ggml_backend_dayan_device_state * device  = nullptr;
    npu_buffer_t *                    storage = nullptr;
    void *                            base    = nullptr;
    size_t                            size    = 0;
    std::vector<npu_buffer_t *>       views;
};

struct ggml_backend_dayan_device_pointer {
    npu_buffer_t * buffer = nullptr;
    uint64_t       offset = 0;
};

static std::mutex                                       ggml_backend_dayan_device_mutex;
static ggml_backend_dayan_device_state *                ggml_backend_dayan_device;
static std::mutex                                       ggml_backend_dayan_buffer_mutex;
static std::vector<ggml_backend_dayan_buffer_context *> ggml_backend_dayan_buffer_contexts;

static void ggml_backend_dayan_device_destroy(ggml_backend_dayan_device_state * device) {
    if (device->stream != nullptr) {
        npu_stream_destroy(device->stream);
    }
    if (device->context != nullptr) {
        npu_context_destroy(device->context);
    }
    if (device->runtime != nullptr) {
        npu_runtime_close(device->runtime);
    }
    delete device;
}

static int ggml_backend_dayan_device_acquire(ggml_backend_dayan_device_state ** result) {
    std::lock_guard<std::mutex> lock(ggml_backend_dayan_device_mutex);
    if (ggml_backend_dayan_device == nullptr) {
        ggml_backend_dayan_device_state * device = new ggml_backend_dayan_device_state;
        npu_runtime_options_t             options;
        npu_runtime_options_init(&options);
        options.default_timeout_ms = GGML_DAYAN_TIMEOUT_MS;

        const char * socket = std::getenv("DAYAN_SERVER_SOCKET");
        if (socket != nullptr && socket[0] != '\0') {
            options.server_socket = socket;
        }

        int status = npu_runtime_open(&device->runtime, &options);
        if (status == 0) {
            status = npu_context_create(device->runtime, nullptr, &device->context);
        }
        if (status == 0) {
            status = npu_stream_create(device->context, nullptr, &device->stream);
        }
        if (status < 0) {
            ggml_backend_dayan_device_destroy(device);
            return status;
        }
        ggml_backend_dayan_device = device;
    }
    ++ggml_backend_dayan_device->references;
    *result = ggml_backend_dayan_device;
    return 0;
}

static void ggml_backend_dayan_device_release(ggml_backend_dayan_device_state * device) {
    std::lock_guard<std::mutex> lock(ggml_backend_dayan_device_mutex);
    GGML_ASSERT(device == ggml_backend_dayan_device && device->references > 0);
    if (--device->references == 0) {
        ggml_backend_dayan_device = nullptr;
        ggml_backend_dayan_device_destroy(device);
    }
}

static bool ggml_backend_dayan_find_device_pointer(const void *                        pointer,
                                                   size_t                              size,
                                                   ggml_backend_dayan_device_pointer * result) {
    const uintptr_t             address = (uintptr_t) pointer;
    std::lock_guard<std::mutex> lock(ggml_backend_dayan_buffer_mutex);
    for (ggml_backend_dayan_buffer_context * context : ggml_backend_dayan_buffer_contexts) {
        const uintptr_t base = (uintptr_t) context->base;
        if (address >= base && address - base <= context->size && size <= context->size - (address - base)) {
            result->buffer = context->storage;
            result->offset = address - base;
            return true;
        }
    }
    return false;
}

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

static int ggml_backend_dayan_copy(npu_stream_t * stream,
                                   npu_buffer_t * destination,
                                   uint64_t       destination_offset,
                                   npu_buffer_t * source,
                                   uint64_t       source_offset,
                                   size_t         size);

static int ggml_backend_dayan_upload(npu_stream_t * stream, npu_buffer_t * buffer, const void * data, size_t size) {
    ggml_backend_dayan_device_pointer source;
    if (ggml_backend_dayan_find_device_pointer(data, size, &source)) {
        return ggml_backend_dayan_copy(stream, buffer, 0, source.buffer, source.offset, size);
    }
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_h2d_async(stream, buffer, 0, data, size, &event);
    return status < 0 ? status : ggml_backend_dayan_event_wait(event);
}

static int ggml_backend_dayan_copy(npu_stream_t * stream,
                                   npu_buffer_t * destination,
                                   uint64_t       destination_offset,
                                   npu_buffer_t * source,
                                   uint64_t       source_offset,
                                   size_t         size) {
    npu_event_t * event = nullptr;
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

static npu_buffer_t * ggml_backend_dayan_tensor_buffer(const ggml_tensor * tensor) {
    return tensor != nullptr ? (npu_buffer_t *) tensor->extra : nullptr;
}

static void ggml_backend_dayan_init_launch_attrs(ggml_backend_dayan_context * ctx,
                                                 const char *                 name,
                                                 npu_kernel_launch_attrs_t *  attrs);
static void ggml_backend_dayan_init_native_launch_attrs(ggml_backend_dayan_context * ctx,
                                                        const char *                 name,
                                                        npu_kernel_launch_attrs_t *  attrs);

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
    const size_t   activation_size  = (size_t) activation_count * sizeof(uint16_t);
    npu_buffer_t * device_buffer    = nullptr;
    int            status           = 0;

    if (activation->type == storage_type) {
        device_buffer = ggml_backend_dayan_tensor_buffer(activation);
        if (device_buffer == nullptr) {
            return -1;
        }
        buffers->activation_by_tensor.emplace(key, device_buffer);
        *result = device_buffer;
        return 0;
    }

    status = ggml_backend_dayan_buffer_create(ctx->context, activation_size, &device_buffer);

    if (status == 0) {
        buffers->activations.push_back(device_buffer);
    }

    if (status == 0 && activation->type == GGML_TYPE_F32 &&
        (storage_type == GGML_TYPE_BF16 || storage_type == GGML_TYPE_F16)) {
        DayanCopyConfig config  = {};
        config.elements         = (uint64_t) activation_count;
        config.source_type      = kDayanDataTypeF32;
        config.destination_type = storage_type == GGML_TYPE_BF16 ? kDayanDataTypeBF16 : kDayanDataTypeF16;
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            config.source_ne[dimension] = (uint32_t) activation->ne[dimension];
            config.source_nb[dimension] = activation->nb[dimension];
        }
        npu_buffer_t * config_buffer = nullptr;
        status                       = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &config_buffer);
        if (status == 0) {
            status = ggml_backend_dayan_upload(ctx->stream, config_buffer, &config, sizeof(config));
        }
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            ggml_backend_dayan_init_native_launch_attrs(ctx, "activation-cast", &attrs);
            status = copy_f32_1c(config_buffer, ggml_backend_dayan_tensor_buffer(activation), device_buffer, &attrs);
        }
        if (config_buffer != nullptr) {
            ggml_backend_dayan_preserve_error(npu_buffer_destroy(config_buffer), &status);
        }
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

static void ggml_backend_dayan_init_native_launch_attrs(ggml_backend_dayan_context * ctx,
                                                        const char *                 name,
                                                        npu_kernel_launch_attrs_t *  attrs) {
    ggml_backend_dayan_init_launch_attrs(ctx, name, attrs);
    attrs->config.task_dim = NPUTaskDim3{ 2, 1, 1 };
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

    const uint64_t          m      = activation->ne[1];
    const uint64_t          n      = weight->ne[1];
    const uint64_t          k      = weight->ne[0];
    const DayanMatmulConfig config = {
        m, n, k, k * sizeof(uint16_t), weight->nb[1], dst->nb[1],
    };
    ggml_backend_dayan_buffers buffers;
    int                        status = 0;

    status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(DayanMatmulPerf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-mul-mat", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     matmul_bf16_1c(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                    ggml_backend_dayan_tensor_buffer(dst), buffers.perf, &attrs) :
                     matmul_fp16_1c(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                    ggml_backend_dayan_tensor_buffer(dst), buffers.perf, &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan matrix multiplication failed with status %d\n", __func__, status);
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
    ggml_backend_dayan_buffers buffers;
    int                        status = 0;

    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(DayanMatmulPerf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-mm-lowp", &attrs);
        status =
            weight->type == GGML_TYPE_BF16 ?
                matmul_quantized_bf16_1c(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                         ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs) :
                matmul_quantized_fp16_1c(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                         ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs);
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
    ggml_backend_dayan_buffers buffers;
    int                        status = 0;

    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(DayanMatmulPerf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(
            ctx, slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ? "ggml-swiglu" : "ggml-geglu", &attrs);

        if (gate->src[0]->type == GGML_TYPE_BF16) {
            status = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ?
                         matmul_swiglu_bf16_1c(buffers.config, activation_buffer,
                                               ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                               ggml_backend_dayan_tensor_buffer(up->src[0]),
                                               ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs) :
                         matmul_geglu_bf16_1c(buffers.config, activation_buffer,
                                              ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                              ggml_backend_dayan_tensor_buffer(up->src[0]),
                                              ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs);
        } else {
            status = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ?
                         matmul_swiglu_fp16_1c(buffers.config, activation_buffer,
                                               ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                               ggml_backend_dayan_tensor_buffer(up->src[0]),
                                               ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs) :
                         matmul_geglu_fp16_1c(buffers.config, activation_buffer,
                                              ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                              ggml_backend_dayan_tensor_buffer(up->src[0]),
                                              ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs);
        }
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan fused matrix multiplication failed with status %d\n", __func__, status);
    }
    return status;
}

static DayanMulMatIdConfig ggml_backend_dayan_mul_mat_id_config(const ggml_tensor * dst) {
    return {
        (uint32_t) dst->src[0]->ne[0], (uint32_t) dst->src[0]->ne[1], (uint32_t) dst->src[0]->ne[2],
        (uint32_t) dst->src[2]->ne[0], (uint32_t) dst->src[2]->ne[1], (uint32_t) dst->src[1]->ne[1],
    };
}

static int ggml_backend_dayan_mul_mat_id(ggml_backend_dayan_context * ctx,
                                         ggml_tensor *                dst,
                                         npu_buffer_t *               activation_buffer) {
    const ggml_tensor *        weight = dst->src[0];
    const DayanMulMatIdConfig  config = ggml_backend_dayan_mul_mat_id_config(dst);
    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(DayanMatmulPerf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-mul-mat-id", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     mul_mat_id_bf16_1c(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                        ggml_backend_dayan_tensor_buffer(dst->src[2]),
                                        ggml_backend_dayan_tensor_buffer(dst), buffers.perf, &attrs) :
                     mul_mat_id_fp16_1c(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                        ggml_backend_dayan_tensor_buffer(dst->src[2]),
                                        ggml_backend_dayan_tensor_buffer(dst), buffers.perf, &attrs);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan expert matrix multiplication failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_mul_mat_id_glu(ggml_backend_dayan_context *          ctx,
                                             const ggml_backend_dayan_graph_slot & slot,
                                             npu_buffer_t *                        activation_buffer) {
    const ggml_tensor *        gate   = slot.nodes[0];
    const ggml_tensor *        up     = slot.nodes[1];
    ggml_tensor *              output = slot.nodes[3];
    const DayanMulMatIdConfig  config = ggml_backend_dayan_mul_mat_id_config(gate);
    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(DayanMatmulPerf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        const bool                swiglu = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_id_swiglu;
        ggml_backend_dayan_init_launch_attrs(ctx, swiglu ? "ggml-moe-swiglu" : "ggml-moe-geglu", &attrs);
        if (gate->src[0]->type == GGML_TYPE_BF16) {
            status =
                swiglu ?
                    mul_mat_id_swiglu_bf16_1c(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs) :
                    mul_mat_id_geglu_bf16_1c(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs);
        } else {
            status =
                swiglu ?
                    mul_mat_id_swiglu_fp16_1c(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs) :
                    mul_mat_id_geglu_fp16_1c(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), buffers.perf, &attrs);
        }
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan fused expert matrix multiplication failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_add_id(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *          output = slot.nodes[0];
    const ggml_tensor *    input  = output->src[0];
    const ggml_tensor *    bias   = output->src[1];
    const DayanAddIdConfig config = {
        (uint32_t) output->ne[0],
        (uint32_t) bias->ne[1],
        (uint32_t) output->ne[1],
        (uint32_t) output->ne[2],
    };
    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "ggml-add-id", &attrs);
        status = add_id_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                               ggml_backend_dayan_tensor_buffer(bias), ggml_backend_dayan_tensor_buffer(output->src[2]),
                               ggml_backend_dayan_tensor_buffer(output), &attrs);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_sort(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *         output = slot.nodes[0];
    const ggml_tensor *   input  = output->src[0];
    const bool            top_k  = slot.kind == ggml_backend_dayan_slot_kind::top_k;
    const DayanSortConfig config = {
        (uint32_t) input->ne[0],
        (uint32_t) ggml_nrows(input),
        (uint32_t) output->ne[0],
        top_k || ggml_get_op_params_i32(output, 0) == GGML_SORT_ORDER_DESC ? kDayanSortOrderDesc : kDayanSortOrderAsc,
    };

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, top_k ? "ggml-top-k" : "ggml-argsort", &attrs);
        status = top_k ? top_k_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                      ggml_backend_dayan_tensor_buffer(output), &attrs) :
                         argsort_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                        ggml_backend_dayan_tensor_buffer(output), &attrs);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static DayanNormConfig ggml_backend_dayan_norm_config(const ggml_tensor * rms) {
    const ggml_tensor * input  = rms->src[0];
    DayanNormConfig     config = {};
    config.width               = (uint32_t) rms->ne[0];
    config.rows                = (uint32_t) ggml_nrows(rms);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.input_ne[dimension] = (uint32_t) input->ne[dimension];
        config.input_nb[dimension] = input->nb[dimension];
    }
    std::memcpy(&config.epsilon_bits, rms->op_params, sizeof(config.epsilon_bits));
    return config;
}

static const ggml_tensor * ggml_backend_dayan_norm_scale(const ggml_tensor * mul, const ggml_tensor * rms) {
    return mul->src[0] == rms ? mul->src[1] : mul->src[0];
}

static int ggml_backend_dayan_elementwise_binary(ggml_backend_dayan_context *          ctx,
                                                 const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *                output = slot.nodes[0];
    const ggml_tensor *          left   = output->src[0];
    const ggml_tensor *          right  = output->src[1];
    const DayanElementwiseConfig config = { (uint64_t) ggml_nelements(output) };
    const size_t                 bytes  = (size_t) config.elements * sizeof(float);
    ggml_backend_dayan_buffers   buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, std::max(sizeof(config), sizeof(DayanRepeatConfig)),
                                                  &buffers.config);
    npu_buffer_t * right_buffer = ggml_backend_dayan_tensor_buffer(right);
    if (status == 0 && !ggml_are_same_shape(right, output)) {
        status                   = ggml_backend_dayan_buffer_create(ctx->context, bytes, &buffers.weight1);
        right_buffer             = buffers.weight1;
        DayanRepeatConfig repeat = {};
        repeat.type              = kDayanDataTypeF32;
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            repeat.source_ne[dimension] = (uint32_t) right->ne[dimension];
            repeat.output_ne[dimension] = (uint32_t) output->ne[dimension];
        }
        if (status == 0) {
            status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &repeat, sizeof(repeat));
        }
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            ggml_backend_dayan_init_native_launch_attrs(ctx, "repeat", &attrs);
            status = repeat_1c(buffers.config, ggml_backend_dayan_tensor_buffer(right), right_buffer, &attrs);
            if (status < 0) {
                GGML_LOG_ERROR("%s: Dayan elementwise repeat failed with status %d\n", __func__, status);
            }
        }
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::elementwise_add:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "elementwise-add", &attrs);
                status = elementwise_add_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(left), right_buffer,
                                                ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_mul:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "elementwise-mul", &attrs);
                status = elementwise_mul_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(left), right_buffer,
                                                ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_sub:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "elementwise-sub", &attrs);
                status = elementwise_sub_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(left), right_buffer,
                                                ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_div:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "elementwise-div", &attrs);
                status = elementwise_div_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(left), right_buffer,
                                                ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            default:
                status = -1;
                break;
        }
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan elementwise operation failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_scale(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanScaleConfig    config = { (uint64_t) ggml_nelements(output), 0, 0 };
    std::memcpy(&config.scale_bits, output->op_params, sizeof(config.scale_bits));
    std::memcpy(&config.bias_bits, (const char *) output->op_params + sizeof(float), sizeof(config.bias_bits));
    if (config.elements == 0) {
        return 0;
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "scale", &attrs);
        status = scale_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                              ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_unary(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *                output = slot.nodes[0];
    const ggml_tensor *          input  = output->src[0];
    const DayanElementwiseConfig config = { (uint64_t) ggml_nelements(output) };
    ggml_backend_dayan_buffers   buffers;
    int status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        const char *              launch_name = nullptr;
        decltype(&sigmoid_f32_1c) kernel      = nullptr;
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
            ggml_backend_dayan_init_native_launch_attrs(ctx, launch_name, &attrs);
            status = kernel(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                            ggml_backend_dayan_tensor_buffer(output), &attrs);
        }
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_reduce(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
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

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::sum:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "sum", &attrs);
                status = sum_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                    ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::sum_rows:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "sum-rows", &attrs);
                status = sum_rows_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                         ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::mean:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "mean", &attrs);
                status = mean_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                     ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            default:
                status = -1;
                break;
        }
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_l2_norm(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *              output = slot.nodes[0];
    const ggml_tensor *        input  = output->src[0];
    DayanNormConfig            config = ggml_backend_dayan_norm_config(output);
    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "l2-norm", &attrs);
        status = l2_norm_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_get_rows(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output  = slot.nodes[0];
    const ggml_tensor * source  = output->src[0];
    const ggml_tensor * indices = output->src[1];
    if (ggml_nelements(output) == 0) {
        return 0;
    }

    uint32_t source_type;
    if (!ggml_backend_dayan_type_code(source->type, &source_type)) {
        return -1;
    }

    DayanRowsConfig config  = {};
    config.row_elements     = (uint32_t) output->ne[0];
    config.rows             = (uint32_t) ggml_nrows(output);
    config.source_rows      = (uint64_t) ggml_nrows(source);
    config.destination_rows = (uint64_t) ggml_nrows(output);
    config.source_type      = source_type;
    config.destination_type = kDayanDataTypeF32;
    config.index_type       = indices->type == GGML_TYPE_I32 ? kDayanDataTypeI32 : kDayanDataTypeI64;
    for (int dimension = 0; dimension < 3; ++dimension) {
        config.index_ne[dimension]       = (uint32_t) indices->ne[dimension];
        config.source_ne[dimension]      = (uint32_t) source->ne[dimension + 1];
        config.destination_ne[dimension] = (uint32_t) output->ne[dimension + 1];
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "get-rows", &attrs);
        status =
            get_rows_1c(buffers.config, ggml_backend_dayan_tensor_buffer(source),
                        ggml_backend_dayan_tensor_buffer(indices), ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_set_rows(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output  = slot.nodes[0];
    const ggml_tensor * source  = output->src[0];
    const ggml_tensor * indices = output->src[1];
    uint32_t            destination_type;
    if (!ggml_backend_dayan_type_code(output->type, &destination_type)) {
        return -1;
    }

    const uint32_t  rows    = (uint32_t) ggml_nrows(source);
    DayanRowsConfig config  = {};
    config.row_elements     = (uint32_t) source->ne[0];
    config.rows             = rows;
    config.source_rows      = rows;
    config.destination_rows = (uint64_t) ggml_nrows(output);
    config.source_type      = kDayanDataTypeF32;
    config.destination_type = destination_type;
    config.index_type       = indices->type == GGML_TYPE_I32 ? kDayanDataTypeI32 : kDayanDataTypeI64;
    config.flags            = kDayanRowsSet;
    for (int dimension = 0; dimension < 3; ++dimension) {
        config.index_ne[dimension]       = (uint32_t) indices->ne[dimension];
        config.source_ne[dimension]      = (uint32_t) source->ne[dimension + 1];
        config.destination_ne[dimension] = (uint32_t) output->ne[dimension + 1];
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_copy(ctx->stream, ggml_backend_dayan_tensor_buffer(output), 0,
                                         ggml_backend_dayan_tensor_buffer(output->src[2]), 0, ggml_nbytes(output));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "set-rows", &attrs);
        status =
            set_rows_1c(buffers.config, ggml_backend_dayan_tensor_buffer(source),
                        ggml_backend_dayan_tensor_buffer(indices), ggml_backend_dayan_tensor_buffer(output), &attrs);
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

    DayanCopyConfig config  = {};
    config.elements         = (uint64_t) ggml_nelements(output);
    config.source_type      = source_type;
    config.destination_type = destination_type;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.source_ne[dimension] = (uint32_t) source->ne[dimension];
        config.source_nb[dimension] = source->nb[dimension];
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "copy", &attrs);
        status = copy_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(source),
                             ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_repeat(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
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

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "repeat", &attrs);
        status = repeat_1c(buffers.config, ggml_backend_dayan_tensor_buffer(source),
                           ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_set(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * base   = output->src[0];
    const ggml_tensor * source = output->src[1];
    DayanSetConfig      config = {
        (uint64_t) ggml_nbytes(output),
        { 0, 0, 0, 0 },
        (uint32_t) ggml_type_size(output->type),
        { (uint32_t) output->op_params[0], (uint32_t) output->op_params[1], (uint32_t) output->op_params[2] },
        (uint32_t) output->op_params[3],
    };
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.source_ne[dimension] = (uint32_t) source->ne[dimension];
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = ggml_backend_dayan_copy(ctx->stream, ggml_backend_dayan_tensor_buffer(output), 0,
                                         ggml_backend_dayan_tensor_buffer(base), 0, ggml_nbytes(output));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "set", &attrs);
        status = set_1c(buffers.config, ggml_backend_dayan_tensor_buffer(base),
                        ggml_backend_dayan_tensor_buffer(source), ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_layout(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
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

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        using kernel_t = int (*)(npu_buffer_t *, npu_buffer_t *, npu_buffer_t *, const npu_kernel_launch_attrs_t *);
        kernel_t     kernel      = nullptr;
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
            ggml_backend_dayan_init_native_launch_attrs(ctx, launch_name, &attrs);
            status = kernel(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                            ggml_backend_dayan_tensor_buffer(output), &attrs);
        }
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_clamp(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output->src[0];
    DayanClampConfig    config = { (uint64_t) ggml_nelements(output), 0, 0 };
    std::memcpy(&config.min_bits, output->op_params, sizeof(config.min_bits));
    std::memcpy(&config.max_bits, (const char *) output->op_params + sizeof(float), sizeof(config.max_bits));

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "clamp", &attrs);
        status = clamp_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                              ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_softmax(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
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
    std::memcpy(&config.max_bias_bits, (const char *) output->op_params + sizeof(float), sizeof(config.max_bias_bits));
    if (mask != nullptr && !ggml_backend_dayan_type_code(mask->type, &config.mask_type)) {
        return -1;
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    npu_buffer_t *             mask_buffer = ggml_backend_dayan_tensor_buffer(mask);
    npu_buffer_t *             sink_buffer = ggml_backend_dayan_tensor_buffer(sinks);
    if (status == 0 && mask_buffer == nullptr) {
        status      = ggml_backend_dayan_buffer_create(ctx->context, sizeof(float), &buffers.weight1);
        mask_buffer = buffers.weight1;
    }
    if (status == 0 && sink_buffer == nullptr) {
        status      = ggml_backend_dayan_buffer_create(ctx->context, sizeof(float), &buffers.input2);
        sink_buffer = buffers.input2;
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "softmax", &attrs);
        status = softmax_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input), mask_buffer, sink_buffer,
                                ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_concat(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * left   = output->src[0];
    const ggml_tensor * right  = output->src[1];
    DayanConcatConfig   config = {};
    config.axis                = (uint32_t) ggml_get_op_params_i32(output, 0);
    if (!ggml_backend_dayan_type_code(output->type, &config.type)) {
        return -1;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.left_ne[dimension]   = (uint32_t) left->ne[dimension];
        config.right_ne[dimension]  = (uint32_t) right->ne[dimension];
        config.output_ne[dimension] = (uint32_t) output->ne[dimension];
        config.left_nb[dimension]   = left->nb[dimension];
        config.right_nb[dimension]  = right->nb[dimension];
        config.output_nb[dimension] = output->nb[dimension];
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "concat", &attrs);
        status = concat_1c(buffers.config, ggml_backend_dayan_tensor_buffer(left),
                           ggml_backend_dayan_tensor_buffer(right), ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_rope(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
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

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    npu_buffer_t *             factor_buffer = ggml_backend_dayan_tensor_buffer(factors);
    std::vector<float>         factor_values;
    if (status == 0 && factor_buffer == nullptr) {
        factor_values.assign((size_t) n_dims / 2, 1.0f);
        status = ggml_backend_dayan_buffer_create(ctx->context, factor_values.size() * sizeof(float), &buffers.input2);
        factor_buffer = buffers.input2;
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0 && factors == nullptr) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input2, factor_values.data(),
                                           factor_values.size() * sizeof(float));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "rope", &attrs);
        status = rope_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                             ggml_backend_dayan_tensor_buffer(positions), factor_buffer,
                             ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_residual_add(ggml_backend_dayan_context *          ctx,
                                           const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *                add    = slot.nodes[0];
    const DayanElementwiseConfig config = {
        (uint64_t) ggml_nelements(add),
    };
    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "residual-add", &attrs);
        status = residual_add_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(add->src[0]),
                                     ggml_backend_dayan_tensor_buffer(add->src[1]),
                                     ggml_backend_dayan_tensor_buffer(add), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan residual add failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_norm_raw(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *         output = slot.nodes[0];
    const bool            rms    = slot.kind == ggml_backend_dayan_slot_kind::rms_norm;
    const DayanNormConfig config = ggml_backend_dayan_norm_config(output);

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, rms ? "rms-norm" : "norm", &attrs);
        status = rms ? rms_norm_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(output->src[0]),
                                       ggml_backend_dayan_tensor_buffer(output), &attrs) :
                       norm_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(output->src[0]),
                                   ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan %s failed with status %d\n", __func__, rms ? "RMSNorm" : "Norm", status);
    }
    return status;
}

static int ggml_backend_dayan_group_norm(ggml_backend_dayan_context * ctx, const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *        output = slot.nodes[0];
    const ggml_tensor *  input  = output->src[0];
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

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "group-norm", &attrs);
        status = group_norm_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                   ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_norm_pipeline(ggml_backend_dayan_context *          ctx,
                                            const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *           add     = slot.nodes[0];
    ggml_tensor *           rms     = slot.nodes[1];
    ggml_tensor *           scaled  = slot.nodes[2];
    ggml_tensor *           lowp    = slot.nodes[3];
    const bool              has_add = add != nullptr;
    const ggml_tensor *     input   = has_add ? add->src[0] : rms->src[0];
    const ggml_tensor *     right   = has_add ? add->src[1] : input;
    const ggml_tensor *     scale   = ggml_backend_dayan_norm_scale(scaled, rms);
    DayanNormPipelineConfig config  = {
        ggml_backend_dayan_norm_config(rms),
        kDayanNormWriteRms | kDayanNormWriteScaled,
        kDayanDataTypeF32,
    };
    if (has_add) {
        config.flags |= kDayanNormHasResidual | kDayanNormWriteResidual;
    }
    if (lowp != nullptr) {
        config.flags |= kDayanNormWriteLowp;
        if (!ggml_backend_dayan_type_code(lowp->type, &config.output_type)) {
            return -1;
        }
    }

    npu_buffer_t * input_buffer    = ggml_backend_dayan_tensor_buffer(input);
    npu_buffer_t * right_buffer    = ggml_backend_dayan_tensor_buffer(right);
    npu_buffer_t * scale_buffer    = ggml_backend_dayan_tensor_buffer(scale);
    npu_buffer_t * residual_buffer = ggml_backend_dayan_tensor_buffer(has_add ? add : rms);
    npu_buffer_t * rms_buffer      = ggml_backend_dayan_tensor_buffer(rms);
    npu_buffer_t * scaled_buffer   = ggml_backend_dayan_tensor_buffer(scaled);
    npu_buffer_t * lowp_buffer     = ggml_backend_dayan_tensor_buffer(lowp != nullptr ? lowp : scaled);
    if (input_buffer == nullptr || right_buffer == nullptr || scale_buffer == nullptr || residual_buffer == nullptr ||
        rms_buffer == nullptr || scaled_buffer == nullptr || lowp_buffer == nullptr) {
        GGML_LOG_ERROR("%s: missing Device buffer for %s -> %s -> %s\n", __func__, input->name, rms->name,
                       scaled->name);
        return -1;
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, has_add ? "add-rms" : "rms-scale", &attrs);
        status = rms_norm_pipeline_1c(buffers.config, input_buffer, right_buffer, scale_buffer, residual_buffer,
                                      rms_buffer, scaled_buffer, lowp_buffer, &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan fused RMSNorm failed with status %d\n", __func__, status);
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
        q->nb[1],
        q->nb[2],
        q->nb[3],
        k->nb[1],
        k->nb[2],
        k->nb[3],
        v->nb[1],
        v->nb[2],
        v->nb[3],
    };
    std::memcpy(&config.scale_bits, &scale, sizeof(scale));
    if (dst->src[3] != nullptr) {
        config.flags = 1U;
    }

    ggml_backend_dayan_buffers buffers;
    int                        status = 0;

    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(int64_t), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-fattn", &attrs);
        status = flash_attention_bf16_1c(buffers.config, ggml_backend_dayan_tensor_buffer(q),
                                         ggml_backend_dayan_tensor_buffer(k), ggml_backend_dayan_tensor_buffer(v),
                                         ggml_backend_dayan_tensor_buffer(dst), buffers.perf, &attrs);
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
    config.q_heads                  = (uint32_t) q->ne[1];
    config.k_heads                  = (uint32_t) k->ne[1];
    config.q_sequences              = (uint32_t) q->ne[3];
    config.k_sequences              = (uint32_t) k->ne[3];
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.q_nb[dimension]    = q->nb[dimension];
        config.k_nb[dimension]    = k->nb[dimension];
        config.v_nb[dimension]    = v->nb[dimension];
        config.g_nb[dimension]    = g->nb[dimension];
        config.beta_nb[dimension] = beta->nb[dimension];
    }
    const float scale = 1.0f / std::sqrt((float) config.head_dim);
    std::memcpy(&config.scale_bits, &scale, sizeof(scale));

    ggml_backend_dayan_buffers buffers;
    int                        status = 0;
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "ggml-gdn", &attrs);
        status = gated_delta_net_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(q),
                                        ggml_backend_dayan_tensor_buffer(k), ggml_backend_dayan_tensor_buffer(v),
                                        ggml_backend_dayan_tensor_buffer(g), ggml_backend_dayan_tensor_buffer(beta),
                                        ggml_backend_dayan_tensor_buffer(state), ggml_backend_dayan_tensor_buffer(dst),
                                        &attrs);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan Gated Delta Net failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_gated_output_projection(ggml_backend_dayan_context *          ctx,
                                                      ggml_backend_dayan_graph_buffers *    graph_buffers,
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
    npu_buffer_t *             qg_lowp = nullptr;
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_activation(ctx, graph_buffers, qg, weight->type, &qg_lowp);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    }
    if (status == 0) {
        status =
            ggml_backend_dayan_buffer_create(ctx->context, (size_t) tokens * width * sizeof(uint16_t), &buffers.input2);
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(DayanMatmulPerf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "gate-out", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     project_gated_output_bf16_1c(buffers.config, ggml_backend_dayan_tensor_buffer(attention), qg_lowp,
                                                  buffers.input2, ggml_backend_dayan_tensor_buffer(weight),
                                                  ggml_backend_dayan_tensor_buffer(projection), buffers.perf, &attrs) :
                     project_gated_output_fp16_1c(buffers.config, ggml_backend_dayan_tensor_buffer(attention), qg_lowp,
                                                  buffers.input2, ggml_backend_dayan_tensor_buffer(weight),
                                                  ggml_backend_dayan_tensor_buffer(projection), buffers.perf, &attrs);
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
    ggml_backend_dayan_buffers buffers;
    int                        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(ctx->context, sizeof(DayanSSMConvPerf), &buffers.perf);
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ssm-conv", &attrs);
        status = ssm_conv_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                 ggml_backend_dayan_tensor_buffer(weight), ggml_backend_dayan_tensor_buffer(output),
                                 buffers.perf, &attrs);
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
    if (ctx->device != nullptr) {
        ggml_backend_dayan_device_release(ctx->device);
    }
    delete ctx;
}

static void ggml_backend_dayan_free(ggml_backend_t backend) {
    ggml_backend_dayan_context_free((ggml_backend_dayan_context *) backend->context);
    delete backend;
}

static void ggml_backend_dayan_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_backend_dayan_buffer_context * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    {
        std::lock_guard<std::mutex> lock(ggml_backend_dayan_buffer_mutex);
        auto                        it =
            std::find(ggml_backend_dayan_buffer_contexts.begin(), ggml_backend_dayan_buffer_contexts.end(), context);
        GGML_ASSERT(it != ggml_backend_dayan_buffer_contexts.end());
        ggml_backend_dayan_buffer_contexts.erase(it);
    }
    for (auto it = context->views.rbegin(); it != context->views.rend(); ++it) {
        npu_buffer_destroy(*it);
    }
    if (context->storage != nullptr) {
        npu_buffer_destroy(context->storage);
    }
    if (context->base != nullptr) {
        munmap(context->base, context->size);
    }
    ggml_backend_dayan_device_release(context->device);
    delete context;
}

static void * ggml_backend_dayan_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_dayan_buffer_context *) buffer->context)->base;
}

static enum ggml_status ggml_backend_dayan_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    const size_t size = ggml_backend_buffer_get_alloc_size(buffer, tensor);
    if (size == 0) {
        tensor->extra = nullptr;
        return GGML_STATUS_SUCCESS;
    }
    ggml_backend_dayan_buffer_context * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    const uintptr_t                     offset  = (uintptr_t) tensor->data - (uintptr_t) context->base;
    if (offset > context->size || size > context->size - offset) {
        return GGML_STATUS_FAILED;
    }

    npu_buffer_t * view = nullptr;
    if (npu_buffer_create_view(context->storage, offset, size, &view) < 0) {
        return GGML_STATUS_FAILED;
    }
    context->views.push_back(view);
    tensor->extra = view;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_dayan_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                 ggml_tensor *         tensor,
                                                 const void *          data,
                                                 size_t                offset,
                                                 size_t                size) {
    if (size == 0) {
        return;
    }
    ggml_backend_dayan_buffer_context * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    npu_buffer_t *                      view    = (npu_buffer_t *) tensor->extra;
    GGML_ASSERT(view != nullptr && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_h2d_async(context->device->stream, view, offset, data, size, &event);
    GGML_ASSERT(status == 0 && ggml_backend_dayan_event_wait(event) == 0);
    if (ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        std::memcpy((char *) tensor->data + offset, data, size);
    }
}

static void ggml_backend_dayan_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                 const ggml_tensor *   tensor,
                                                 void *                data,
                                                 size_t                offset,
                                                 size_t                size) {
    if (size == 0) {
        return;
    }
    ggml_backend_dayan_buffer_context * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    npu_buffer_t *                      view    = (npu_buffer_t *) tensor->extra;
    GGML_ASSERT(view != nullptr && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_d2h_async(context->device->stream, data, view, offset, size, &event);
    GGML_ASSERT(status == 0 && ggml_backend_dayan_event_wait(event) == 0);
    if (data != (char *) tensor->data + offset) {
        std::memcpy((char *) tensor->data + offset, data, size);
    }
}

static void ggml_backend_dayan_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    uint8_t               value,
                                                    size_t                offset,
                                                    size_t                size) {
    if (size == 0) {
        return;
    }
    ggml_backend_dayan_buffer_context * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    npu_buffer_t *                      view    = (npu_buffer_t *) tensor->extra;
    GGML_ASSERT(view != nullptr && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    std::vector<uint8_t> chunk(std::min<size_t>(size, 1024U * 1024U), value);
    for (size_t copied = 0; copied < size; copied += chunk.size()) {
        const size_t  current = std::min(chunk.size(), size - copied);
        npu_event_t * event   = nullptr;
        const int     status =
            npu_memcpy_h2d_async(context->device->stream, view, offset + copied, chunk.data(), current, &event);
        GGML_ASSERT(status == 0 && ggml_backend_dayan_event_wait(event) == 0);
    }
}

static bool ggml_backend_dayan_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                 const ggml_tensor *   src,
                                                 ggml_tensor *         dst) {
    if (ggml_nbytes(dst) == 0) {
        return true;
    }
    if (src->buffer == nullptr || src->buffer->buft != buffer->buft || src->extra == nullptr || dst->extra == nullptr) {
        return false;
    }
    ggml_backend_dayan_buffer_context * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    return ggml_backend_dayan_copy(context->device->stream, (npu_buffer_t *) dst->extra, 0, (npu_buffer_t *) src->extra,
                                   0, ggml_nbytes(dst)) == 0;
}

static void ggml_backend_dayan_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_dayan_buffer_context * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    std::vector<uint8_t>                chunk(std::min<size_t>(context->size, 1024U * 1024U), value);
    for (size_t copied = 0; copied < context->size; copied += chunk.size()) {
        const size_t  current = std::min(chunk.size(), context->size - copied);
        npu_event_t * event   = nullptr;
        const int     status =
            npu_memcpy_h2d_async(context->device->stream, context->storage, copied, chunk.data(), current, &event);
        GGML_ASSERT(status == 0 && ggml_backend_dayan_event_wait(event) == 0);
    }
}

static const ggml_backend_buffer_i ggml_backend_dayan_buffer_i = {
    /* .free_buffer   = */ ggml_backend_dayan_buffer_free,
    /* .get_base      = */ ggml_backend_dayan_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_dayan_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_dayan_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_dayan_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_dayan_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_dayan_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_dayan_buffer_clear,
    /* .reset         = */ nullptr,
};

static const char * ggml_backend_dayan_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_DAYAN_NAME;
}

static ggml_backend_buffer_t ggml_backend_dayan_buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    if (size == 0) {
        return ggml_backend_buffer_init(buft, {}, nullptr, 0);
    }

    ggml_backend_dayan_device_state * device = nullptr;
    int                               status = ggml_backend_dayan_device_acquire(&device);
    if (status < 0) {
        GGML_LOG_ERROR("%s: failed to initialize the Dayan runtime with status %d\n", __func__, status);
        return nullptr;
    }

    ggml_backend_dayan_buffer_context * context = new ggml_backend_dayan_buffer_context;
    context->device                             = device;
    context->size                               = size;
    context->base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (context->base == MAP_FAILED) {
        context->base = nullptr;
        status        = -1;
    }
    if (status == 0) {
        status = ggml_backend_dayan_buffer_create(device->context, size, &context->storage);
    }
    if (status < 0) {
        if (context->base != nullptr) {
            munmap(context->base, size);
        }
        ggml_backend_dayan_device_release(device);
        delete context;
        GGML_LOG_ERROR("%s: failed to allocate %zu Device bytes with status %d\n", __func__, size, status);
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(ggml_backend_dayan_buffer_mutex);
        ggml_backend_dayan_buffer_contexts.push_back(context);
    }
    return ggml_backend_buffer_init(buft, ggml_backend_dayan_buffer_i, context, size);
}

static size_t ggml_backend_dayan_buffer_type_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

static const ggml_backend_buffer_type_i ggml_backend_dayan_buffer_type_i = {
    /* .get_name       = */ ggml_backend_dayan_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_dayan_buffer_type_alloc,
    /* .get_alignment  = */ ggml_backend_dayan_buffer_type_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_dayan_buffer_type(ggml_backend_dev_t device) {
    static ggml_backend_buffer_type buffer_type = {
        /* .iface   = */ ggml_backend_dayan_buffer_type_i,
        /* .device  = */ device,
        /* .context = */ nullptr,
    };
    return &buffer_type;
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
            case ggml_backend_dayan_slot_kind::mul_mat_id:
            case ggml_backend_dayan_slot_kind::mul_mat_id_pair:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    for (size_t i = 0; status == 0 && i < slot.node_count; ++i) {
                        status = ggml_backend_dayan_mul_mat_id(ctx, slot.nodes[i], activation_buffer);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::mul_mat_id_swiglu:
            case ggml_backend_dayan_slot_kind::mul_mat_id_geglu:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat_id_glu(ctx, slot, activation_buffer);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::add_id:
                status = ggml_backend_dayan_add_id(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::argsort:
            case ggml_backend_dayan_slot_kind::top_k:
                status = ggml_backend_dayan_sort(ctx, slot);
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
            case ggml_backend_dayan_slot_kind::norm_pipeline:
                status = ggml_backend_dayan_norm_pipeline(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::flash_attention:
                status = ggml_backend_dayan_flash_attention(ctx, slot.nodes[0]);
                break;
            case ggml_backend_dayan_slot_kind::gated_delta_net:
                status = ggml_backend_dayan_gated_delta_net(ctx, slot.nodes[0]);
                break;
            case ggml_backend_dayan_slot_kind::gated_output_projection:
                status = ggml_backend_dayan_gated_output_projection(ctx, &graph_buffers, slot);
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
            "%s: lowered %d nodes to %zu slots, paired=%zu, fused=%zu, residual=%zu, rms=%zu, "
            "attention=%zu, delta=%zu, gated=%zu, conv=%zu, moe=%zu, moe-fused=%zu, routing=%zu, activations=%zu\n",
            __func__, cgraph->n_nodes, plan.slots.size(), plan.paired_mul_mat_slots, plan.fused_mul_mat_slots,
            plan.residual_add_slots, plan.rms_norm_slots, plan.flash_attention_slots, plan.gated_delta_net_slots,
            plan.gated_output_slots, plan.ssm_conv_slots, plan.moe_slots, plan.fused_moe_slots, plan.routing_slots,
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
    ggml_backend_dayan_context * ctx    = new ggml_backend_dayan_context;
    int                          status = ggml_backend_dayan_device_acquire(&ctx->device);
    if (status < 0) {
        GGML_LOG_ERROR("%s: failed to initialize the Dayan runtime with status %d\n", __func__, status);
        ggml_backend_dayan_context_free(ctx);
        return nullptr;
    }
    ctx->runtime = ctx->device->runtime;
    ctx->context = ctx->device->context;
    ctx->stream  = ctx->device->stream;

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
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_dayan_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_dayan_init();
}

static ggml_backend_buffer_type_t ggml_backend_dayan_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_dayan_buffer_type(dev);
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
        case GGML_OP_MUL_MAT_ID:
            return ggml_backend_dayan_supports_mul_mat_id(op);
        case GGML_OP_GLU:
            return ggml_backend_dayan_supports_glu(op) || ggml_backend_dayan_supports_mul_mat_id_glu(op);
        case GGML_OP_CPY:
            return ggml_backend_dayan_supports_quantized_glu(op) || ggml_backend_dayan_supports_quantized_mul_mat(op) ||
                   ggml_backend_dayan_supports_quantized_rms_norm(op) || ggml_backend_dayan_supports_copy(op);
        case GGML_OP_CLAMP:
            return ggml_backend_dayan_supports_clamp(op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_dayan_supports_softmax(op);
        case GGML_OP_GET_ROWS:
            return ggml_backend_dayan_supports_get_rows(op);
        case GGML_OP_ADD_ID:
            return ggml_backend_dayan_supports_add_id(op);
        case GGML_OP_ARGSORT:
            return ggml_backend_dayan_supports_argsort(op);
        case GGML_OP_TOP_K:
            return ggml_backend_dayan_supports_top_k(op);
        case GGML_OP_SET_ROWS:
            return ggml_backend_dayan_supports_set_rows(op);
        case GGML_OP_CONCAT:
            return ggml_backend_dayan_supports_concat(op);
        case GGML_OP_ROPE:
            return ggml_backend_dayan_supports_rope(op);
        case GGML_OP_ADD:
            return ggml_backend_dayan_supports_residual_add(op) || ggml_backend_dayan_supports_elementwise_add(op);
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
    return buft == ggml_backend_dayan_buffer_type(dev);
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
    /* .buffer_from_host_ptr = */ nullptr,
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
