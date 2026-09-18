#include "ggml-dayan.h"

#include "ggml-backend-impl.h"
#include "ggml-dayan-graph.h"
#include "ggml-dayan-kernels.h"
#include "ggml-dayan-mega.h"
#include "ggml-dayan-mxfp4-layout.h"
#include "ggml-impl.h"

#include <sys/mman.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
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
    DayanTopology   topology;
};

struct ggml_backend_dayan_slot_resources {
    npu_buffer_t *       config             = nullptr;
    size_t               config_capacity    = 0;
    std::vector<uint8_t> config_data;
    npu_buffer_t *       result             = nullptr;
    size_t               result_capacity    = 0;
    npu_buffer_t *       workspace          = nullptr;
    size_t               workspace_capacity = 0;
};

struct ggml_backend_dayan_context {
    ggml_backend_dayan_device_state *              device  = nullptr;
    npu_runtime_t *                                runtime = nullptr;
    npu_context_t *                                context = nullptr;
    npu_stream_t *                                 stream  = nullptr;
    std::vector<ggml_backend_dayan_slot_resources> slot_resources;
    size_t                                         active_slot = SIZE_MAX;
};

struct ggml_backend_dayan_mxfp4_extra {
    npu_buffer_t * data;
    npu_buffer_t * scales;
    DayanMxfp4Layout layout;
};

struct ggml_backend_dayan_buffer_context {
    ggml_backend_dayan_device_state * device  = nullptr;
    npu_buffer_t *                    storage = nullptr;
    void *                            base    = nullptr;
    size_t                            size    = 0;
    std::vector<npu_buffer_t *>       views;
    std::vector<ggml_backend_dayan_mxfp4_extra *> mxfp4;
};

struct ggml_backend_dayan_device_pointer {
    npu_buffer_t * buffer = nullptr;
    uint64_t       offset = 0;
};

static std::mutex                                       ggml_backend_dayan_device_mutex;
static ggml_backend_dayan_device_state *                ggml_backend_dayan_device;
static std::mutex                                       ggml_backend_dayan_buffer_mutex;
static std::vector<ggml_backend_dayan_buffer_context *> ggml_backend_dayan_buffer_contexts;
static std::mutex                                       ggml_backend_dayan_trace_mutex;
static ggml_backend_dayan_slot_trace_callback           ggml_backend_dayan_trace_callback;
static void *                                           ggml_backend_dayan_trace_user_data;
static uint64_t                                         ggml_backend_dayan_mega_launch_count;

static uint64_t ggml_backend_dayan_get_mega_launch_count() {
    return ggml_backend_dayan_mega_launch_count;
}

static ggml_backend_dayan_slot_filter_callback          ggml_backend_dayan_filter_callback;
static void *                                           ggml_backend_dayan_filter_user_data;

void ggml_backend_dayan_set_slot_trace_callback(ggml_backend_dayan_slot_trace_callback callback, void * user_data) {
    std::lock_guard<std::mutex> lock(ggml_backend_dayan_trace_mutex);
    ggml_backend_dayan_trace_callback  = callback;
    ggml_backend_dayan_trace_user_data = user_data;
}

void ggml_backend_dayan_set_slot_filter_callback(ggml_backend_dayan_slot_filter_callback callback, void * user_data) {
    std::lock_guard<std::mutex> lock(ggml_backend_dayan_trace_mutex);
    ggml_backend_dayan_filter_callback  = callback;
    ggml_backend_dayan_filter_user_data = user_data;
}

static const char * ggml_backend_dayan_slot_kind_name(ggml_backend_dayan_slot_kind kind) {
    switch (kind) {
        case ggml_backend_dayan_slot_kind::mul_mat:
            return "mul_mat";
        case ggml_backend_dayan_slot_kind::mul_mat_silu:
            return "mul_mat_silu";
        case ggml_backend_dayan_slot_kind::mul_mat_quantized:
            return "mul_mat_quantized";
        case ggml_backend_dayan_slot_kind::mul_mat_pair:
            return "mul_mat_pair";
        case ggml_backend_dayan_slot_kind::mul_mat_swiglu:
            return "mul_mat_swiglu";
        case ggml_backend_dayan_slot_kind::mul_mat_geglu:
            return "mul_mat_geglu";
        case ggml_backend_dayan_slot_kind::mul_mat_id:
            return "mul_mat_id";
        case ggml_backend_dayan_slot_kind::mul_mat_id_pair:
            return "mul_mat_id_pair";
        case ggml_backend_dayan_slot_kind::mul_mat_id_swiglu:
            return "mul_mat_id_swiglu";
        case ggml_backend_dayan_slot_kind::mul_mat_id_geglu:
            return "mul_mat_id_geglu";
        case ggml_backend_dayan_slot_kind::add_id:
            return "add_id";
        case ggml_backend_dayan_slot_kind::argsort:
            return "argsort";
        case ggml_backend_dayan_slot_kind::top_k:
            return "top_k";
        case ggml_backend_dayan_slot_kind::elementwise_add:
            return "elementwise_add";
        case ggml_backend_dayan_slot_kind::elementwise_mul:
            return "elementwise_mul";
        case ggml_backend_dayan_slot_kind::elementwise_sub:
            return "elementwise_sub";
        case ggml_backend_dayan_slot_kind::elementwise_div:
            return "elementwise_div";
        case ggml_backend_dayan_slot_kind::scale:
            return "scale";
        case ggml_backend_dayan_slot_kind::sigmoid:
            return "sigmoid";
        case ggml_backend_dayan_slot_kind::silu:
            return "silu";
        case ggml_backend_dayan_slot_kind::softplus:
            return "softplus";
        case ggml_backend_dayan_slot_kind::unary_math:
            return "unary_math";
        case ggml_backend_dayan_slot_kind::sum:
            return "sum";
        case ggml_backend_dayan_slot_kind::sum_rows:
            return "sum_rows";
        case ggml_backend_dayan_slot_kind::mean:
            return "mean";
        case ggml_backend_dayan_slot_kind::l2_norm:
            return "l2_norm";
        case ggml_backend_dayan_slot_kind::get_rows:
            return "get_rows";
        case ggml_backend_dayan_slot_kind::set_rows:
            return "set_rows";
        case ggml_backend_dayan_slot_kind::copy:
            return "copy";
        case ggml_backend_dayan_slot_kind::repeat:
            return "repeat";
        case ggml_backend_dayan_slot_kind::set:
            return "set";
        case ggml_backend_dayan_slot_kind::diag:
            return "diag";
        case ggml_backend_dayan_slot_kind::diag_mask_inf:
            return "diag_mask_inf";
        case ggml_backend_dayan_slot_kind::diag_mask_zero:
            return "diag_mask_zero";
        case ggml_backend_dayan_slot_kind::pad:
            return "pad";
        case ggml_backend_dayan_slot_kind::pad_circular:
            return "pad_circular";
        case ggml_backend_dayan_slot_kind::pad_reflect_1d:
            return "pad_reflect_1d";
        case ggml_backend_dayan_slot_kind::roll:
            return "roll";
        case ggml_backend_dayan_slot_kind::clamp:
            return "clamp";
        case ggml_backend_dayan_slot_kind::softmax:
            return "softmax";
        case ggml_backend_dayan_slot_kind::concat:
            return "concat";
        case ggml_backend_dayan_slot_kind::rope:
            return "rope";
        case ggml_backend_dayan_slot_kind::residual_add:
            return "residual_add";
        case ggml_backend_dayan_slot_kind::rms_norm:
            return "rms_norm";
        case ggml_backend_dayan_slot_kind::norm:
            return "norm";
        case ggml_backend_dayan_slot_kind::group_norm:
            return "group_norm";
        case ggml_backend_dayan_slot_kind::norm_pipeline:
            return "norm_pipeline";
        case ggml_backend_dayan_slot_kind::flash_attention:
            return "flash_attention";
        case ggml_backend_dayan_slot_kind::gated_delta_net:
            return "gated_delta_net";
        case ggml_backend_dayan_slot_kind::gated_output_projection:
            return "gated_output_projection";
        case ggml_backend_dayan_slot_kind::ssm_conv:
            return "ssm_conv";
        case ggml_backend_dayan_slot_kind::ssm_conv_silu:
            return "ssm_conv_silu";
    }
    return "unknown";
}

static size_t ggml_backend_dayan_slot_outputs(const ggml_backend_dayan_graph_slot & slot,
                                              const ggml_tensor **                  outputs) {
    switch (slot.kind) {
        case ggml_backend_dayan_slot_kind::mul_mat_silu:
            outputs[0] = slot.nodes[1];
            return 1;
        case ggml_backend_dayan_slot_kind::mul_mat_pair:
        case ggml_backend_dayan_slot_kind::mul_mat_id_pair:
            outputs[0] = slot.nodes[0];
            outputs[1] = slot.nodes[1];
            return 2;
        case ggml_backend_dayan_slot_kind::mul_mat_quantized:
            outputs[0] = slot.nodes[1];
            return 1;
        case ggml_backend_dayan_slot_kind::mul_mat_swiglu:
        case ggml_backend_dayan_slot_kind::mul_mat_geglu:
        case ggml_backend_dayan_slot_kind::mul_mat_id_swiglu:
        case ggml_backend_dayan_slot_kind::mul_mat_id_geglu:
            outputs[0] = slot.nodes[3];
            return 1;
        case ggml_backend_dayan_slot_kind::norm_pipeline: {
            size_t count = 0;
            for (size_t index = 0; index < 4; ++index) {
                if ((slot.output_mask & (1U << index)) != 0U) {
                    outputs[count++] = slot.nodes[index];
                }
            }
            return count;
        }
        case ggml_backend_dayan_slot_kind::gated_output_projection:
            outputs[0] = slot.nodes[2];
            return 1;
        case ggml_backend_dayan_slot_kind::ssm_conv_silu:
            outputs[0] = slot.nodes[1];
            return 1;
        default:
            outputs[0] = slot.nodes[0];
            return 1;
    }
}

struct ggml_backend_dayan_slot_implementation {
    const char * executor;
    const char * launch_name;
    std::string  kernel;
    const char * task_mode;
    uint32_t     task_dim_x;
    uint32_t     task_dim_y = 1;
};

static bool ggml_backend_dayan_is_mxfp4_slot(const ggml_backend_dayan_graph_slot & slot);

static ggml_backend_dayan_slot_implementation
ggml_backend_dayan_slot_implementation_info(const ggml_backend_dayan_graph_slot & slot, const DayanTopology & topology) {
    if (ggml_backend_dayan_is_mxfp4_slot(slot)) {
        const bool copy = slot.kind == ggml_backend_dayan_slot_kind::copy;
        return {"ggml_backend_dayan_mxfp4_slot", copy ? "quantize-mxfp4" : "matmul-mxfp4",
                copy ? "quantize_mxfp4" : topology.clusters ? "matmul_mxfp4_4c" : "matmul_mxfp4_1c",
                copy || !topology.clusters ? "single-core" : "four-core", copy || !topology.clusters ? topology.cores : 4,
                copy || !topology.clusters ? 1 : topology.clusters};
    }
    const bool four_core = slot.kind == ggml_backend_dayan_slot_kind::mul_mat ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_silu ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_quantized ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_pair ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_geglu ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_id ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_id_pair ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_id_swiglu ||
                           slot.kind == ggml_backend_dayan_slot_kind::mul_mat_id_geglu ||
                           slot.kind == ggml_backend_dayan_slot_kind::gated_output_projection;
    ggml_backend_dayan_slot_implementation result = {
        "unknown", "unknown", "unknown", four_core ? "four-core" : "single-core",
        four_core ? 4U : kDayanNativeMaxTasks,
    };
    const ggml_tensor * node = slot.nodes[0];

    switch (slot.kind) {
        case ggml_backend_dayan_slot_kind::mul_mat:
        case ggml_backend_dayan_slot_kind::mul_mat_silu: {
            const bool bf16 = node->src[0]->type == GGML_TYPE_BF16;
            const bool silu = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_silu;
            const bool inference = (node->src[0]->ne[2] == 1 && node->src[0]->ne[3] == 1 ?
                                    ggml_nrows(node->src[1]) : node->src[1]->ne[1]) <= 512;
            result.executor = "ggml_backend_dayan_mul_mat";
            result.launch_name = silu ? "ggml-mm-silu" : "ggml-mul-mat";
            result.kernel = silu ? (bf16 ? "matmul_silu_bf16_4c" : "matmul_silu_fp16_4c") :
                                   (bf16 ? "matmul_bf16_4c" : "matmul_fp16_4c");
            if (inference) {
                result.kernel = silu ? (bf16 ? "matmul_silu_inference_bf16_4c" : "matmul_silu_inference_fp16_4c") :
                                       (bf16 ? "matmul_inference_bf16_4c" : "matmul_inference_fp16_4c");
            }
        } break;
        case ggml_backend_dayan_slot_kind::mul_mat_quantized: {
            const bool bf16 = node->src[0]->type == GGML_TYPE_BF16;
            result.executor    = "ggml_backend_dayan_mul_mat_quantized";
            result.launch_name = "ggml-mm-lowp";
            result.kernel      = bf16 ? "matmul_quantized_bf16_4c" : "matmul_quantized_fp16_4c";
            if (node->src[1]->ne[1] <= 512) {
                result.kernel = bf16 ? "matmul_quantized_inference_bf16_4c" : "matmul_quantized_inference_fp16_4c";
            }
        } break;
        case ggml_backend_dayan_slot_kind::mul_mat_pair: {
            const bool bf16 = node->src[0]->type == GGML_TYPE_BF16;
            result.executor    = "ggml_backend_dayan_mul_mat_pair";
            result.launch_name = "ggml-mm-pair";
            result.kernel      = bf16 ? "matmul_pair_bf16_4c" : "matmul_pair_fp16_4c";
        } break;
        case ggml_backend_dayan_slot_kind::mul_mat_swiglu:
        case ggml_backend_dayan_slot_kind::mul_mat_geglu: {
            const bool bf16    = node->src[0]->type == GGML_TYPE_BF16;
            const bool swiglu  = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu;
            result.executor    = "ggml_backend_dayan_mul_mat_glu";
            result.launch_name = swiglu ? "ggml-swiglu" : "ggml-geglu";
            result.kernel      = swiglu ? (bf16 ? "matmul_swiglu_bf16_4c" : "matmul_swiglu_fp16_4c") :
                                          (bf16 ? "matmul_geglu_bf16_4c" : "matmul_geglu_fp16_4c");
            if (node->src[1]->ne[1] <= 512) {
                result.kernel = swiglu ? (bf16 ? "matmul_swiglu_inference_bf16_4c" : "matmul_swiglu_inference_fp16_4c") :
                                        (bf16 ? "matmul_geglu_inference_bf16_4c" : "matmul_geglu_inference_fp16_4c");
            }
        } break;
        case ggml_backend_dayan_slot_kind::mul_mat_id: {
            const bool bf16 = node->src[0]->type == GGML_TYPE_BF16;
            result.executor    = "ggml_backend_dayan_mul_mat_id";
            result.launch_name = "ggml-mul-mat-id";
            result.kernel      = bf16 ? "mul_mat_id_bf16_4c" : "mul_mat_id_fp16_4c";
        } break;
        case ggml_backend_dayan_slot_kind::mul_mat_id_pair: {
            const bool bf16 = node->src[0]->type == GGML_TYPE_BF16;
            result.executor    = "ggml_backend_dayan_mul_mat_id_pair";
            result.launch_name = "ggml-moe-pair";
            result.kernel      = bf16 ? "mul_mat_id_pair_bf16_4c" : "mul_mat_id_pair_fp16_4c";
        } break;
        case ggml_backend_dayan_slot_kind::mul_mat_id_swiglu:
        case ggml_backend_dayan_slot_kind::mul_mat_id_geglu: {
            const bool bf16    = node->src[0]->type == GGML_TYPE_BF16;
            const bool swiglu  = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_id_swiglu;
            result.executor    = "ggml_backend_dayan_mul_mat_id_glu";
            result.launch_name = swiglu ? "ggml-moe-swiglu" : "ggml-moe-geglu";
            result.kernel      = swiglu ? (bf16 ? "mul_mat_id_swiglu_bf16_4c" : "mul_mat_id_swiglu_fp16_4c") :
                                          (bf16 ? "mul_mat_id_geglu_bf16_4c" : "mul_mat_id_geglu_fp16_4c");
        } break;
        case ggml_backend_dayan_slot_kind::add_id:
            result = { "ggml_backend_dayan_add_id", "ggml-add-id", "add_id_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::argsort:
            result = { "ggml_backend_dayan_sort", slot.sort_k < node->ne[0] ? "ggml-sort-topk" : "ggml-argsort", "argsort_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::top_k:
            result = { "ggml_backend_dayan_sort", "ggml-top-k", "top_k_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::elementwise_add:
            result = { "ggml_backend_dayan_elementwise_binary", "elementwise-add", "elementwise_add_f32_1c",
                       "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::elementwise_mul:
            result = { "ggml_backend_dayan_elementwise_binary", "elementwise-mul", "elementwise_mul_f32_1c",
                       "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::elementwise_sub:
            result = { "ggml_backend_dayan_elementwise_binary", "elementwise-sub", "elementwise_sub_f32_1c",
                       "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::elementwise_div:
            result = { "ggml_backend_dayan_elementwise_binary", "elementwise-div", "elementwise_div_f32_1c",
                       "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::scale:
            result = { "ggml_backend_dayan_scale", "scale", "scale_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::sigmoid:
            result = { "ggml_backend_dayan_unary", "sigmoid", "sigmoid_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::silu:
            result = { "ggml_backend_dayan_unary", "silu", "silu_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::softplus:
            result = { "ggml_backend_dayan_unary", "softplus", "softplus_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::unary_math:
            result.executor = "ggml_backend_dayan_unary";
            if (node->op == GGML_OP_SQR) {
                result.launch_name = "square";
                result.kernel      = "square_f32_1c";
            } else if (node->op == GGML_OP_SQRT) {
                result.launch_name = "sqrt";
                result.kernel      = "sqrt_f32_1c";
            } else if (node->op == GGML_OP_LOG) {
                result.launch_name = "log";
                result.kernel      = "log_f32_1c";
            } else if (node->op == GGML_OP_SIN) {
                result.launch_name = "sin";
                result.kernel      = "sin_f32_1c";
            } else if (node->op == GGML_OP_COS) {
                result.launch_name = "cos";
                result.kernel      = "cos_f32_1c";
            } else if (node->op == GGML_OP_UNARY) {
                switch (ggml_get_unary_op(node)) {
                    case GGML_UNARY_OP_ABS:
                        result.launch_name = "abs";
                        result.kernel      = "abs_f32_1c";
                        break;
                    case GGML_UNARY_OP_NEG:
                        result.launch_name = "neg";
                        result.kernel      = "neg_f32_1c";
                        break;
                    case GGML_UNARY_OP_TANH:
                        result.launch_name = "tanh";
                        result.kernel      = "tanh_f32_1c";
                        break;
                    case GGML_UNARY_OP_RELU:
                        result.launch_name = "relu";
                        result.kernel      = "relu_f32_1c";
                        break;
                    case GGML_UNARY_OP_GELU:
                        result.launch_name = "gelu";
                        result.kernel      = "gelu_f32_1c";
                        break;
                    case GGML_UNARY_OP_GELU_QUICK:
                        result.launch_name = "gelu-quick";
                        result.kernel      = "gelu_quick_f32_1c";
                        break;
                    case GGML_UNARY_OP_EXP:
                        result.launch_name = "exp";
                        result.kernel      = "exp_f32_1c";
                        break;
                    default:
                        break;
                }
            }
            break;
        case ggml_backend_dayan_slot_kind::sum:
            result = { "ggml_backend_dayan_reduce", "sum", "sum_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::sum_rows:
            result = { "ggml_backend_dayan_reduce", "sum-rows", "sum_rows_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::mean:
            result = { "ggml_backend_dayan_reduce", "mean", "mean_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::l2_norm:
            result = { "ggml_backend_dayan_l2_norm", "l2-norm", "l2_norm_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::get_rows:
            result = { "ggml_backend_dayan_get_rows", "get-rows", "get_rows_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::set_rows:
            result = { "ggml_backend_dayan_set_rows", "set-rows", "set_rows_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::copy:
            result = { "ggml_backend_dayan_copy_tensor", "copy", "copy_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::repeat:
            result = { "ggml_backend_dayan_repeat", "repeat", "repeat_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::set:
            result = { "ggml_backend_dayan_set", "set", "set_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::diag:
            result = { "ggml_backend_dayan_layout", "diag", "diag_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::diag_mask_inf:
            result = { "ggml_backend_dayan_layout", "diag-mask-inf", "diag_mask_inf_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::diag_mask_zero:
            result = { "ggml_backend_dayan_layout", "diag-mask-zero", "diag_mask_zero_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::pad:
            result = { "ggml_backend_dayan_layout", "pad", "pad_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::pad_circular:
            result = { "ggml_backend_dayan_layout", "pad-circular", "pad_circular_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::pad_reflect_1d:
            result = { "ggml_backend_dayan_layout", "pad-reflect-1d", "pad_reflect_1d_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::roll:
            result = { "ggml_backend_dayan_layout", "roll", "roll_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::clamp:
            result = { "ggml_backend_dayan_clamp", "clamp", "clamp_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::softmax:
            result = { "ggml_backend_dayan_softmax", "softmax", "softmax_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::concat:
            result = { "ggml_backend_dayan_concat", "concat", "concat_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::rope:
            result = { "ggml_backend_dayan_rope", "rope", "rope_f32_1c", "single-core", kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::residual_add:
            result.executor    = "ggml_backend_dayan_residual_add";
            result.launch_name = "residual-add";
            result.kernel      = node->type == GGML_TYPE_F32 ? "residual_add_f32_1c" :
                                 node->type == GGML_TYPE_BF16 ? "residual_add_bf16_1c" :
                                                               "residual_add_fp16_1c";
            break;
        case ggml_backend_dayan_slot_kind::rms_norm:
            result = { "ggml_backend_dayan_norm_raw", "rms-norm", "rms_norm_f32_1c", "single-core",
                       (uint32_t) std::min<int64_t>(ggml_nrows(node), kDayanNativeMaxTasks) };
            break;
        case ggml_backend_dayan_slot_kind::norm:
            result = { "ggml_backend_dayan_norm_raw", "norm", "norm_f32_1c", "single-core",
                       (uint32_t) std::min<int64_t>(ggml_nrows(node), kDayanNativeMaxTasks) };
            break;
        case ggml_backend_dayan_slot_kind::group_norm:
            result = { "ggml_backend_dayan_group_norm", "group-norm", "group_norm_f32_1c", "single-core",
                       kDayanNativeMaxTasks };
            break;
        case ggml_backend_dayan_slot_kind::norm_pipeline:
            result = { "ggml_backend_dayan_norm_pipeline", slot.nodes[0] != nullptr ? "add-rms" : "rms-scale",
                       "rms_norm_pipeline_1c", "single-core",
                       (uint32_t) std::min<int64_t>(ggml_nrows(slot.nodes[1]), kDayanNativeMaxTasks) };
            break;
        case ggml_backend_dayan_slot_kind::flash_attention: {
            const bool bf16 = node->src[0]->type == GGML_TYPE_BF16;
            uint32_t causal_offset = 0;
            uint32_t kv_len = 0;
            const bool split_kv = topology.clusters && ggml_backend_dayan_flash_attention_lengths(node, &causal_offset, &kv_len) &&
                                  dayan_flash_attention_use_split_kv(node->src[0]->ne[1], kv_len);
            const bool four_core = topology.clusters && (node->src[0]->ne[1] > 64 || split_kv);
            result = { "ggml_backend_dayan_flash_attention", bf16 ? "flash-attn-bf16" : "flash-attn-fp16",
                       split_kv ? (bf16 ? "flash_attention_decode_bf16_4c" : "flash_attention_decode_fp16_4c") :
                       four_core ? (bf16 ? "flash_attention_bf16_4c" : "flash_attention_fp16_4c") :
                                   (bf16 ? "flash_attention_bf16_1c" : "flash_attention_fp16_1c"),
                       four_core ? "four-core" : "single-core", kDayanNativeMaxTasks };
        } break;
        case ggml_backend_dayan_slot_kind::gated_delta_net:
            result = { "ggml_backend_dayan_gated_delta_net", "ggml-gdn",
                       node->src[3]->ne[0] == 1 && ggml_get_op_params_i32(node, 0) == 1 && node->src[2]->ne[2] >= 16 ?
                           "gated_delta_net_chunk_f32_1c" : "gated_delta_net_f32_1c", "single-core",
                       (uint32_t) std::min<uint64_t>(node->src[2]->ne[1] * node->src[2]->ne[3], kDayanGatedDeltaNetMaxTasks) };
            break;
        case ggml_backend_dayan_slot_kind::gated_output_projection: {
            const ggml_tensor * projection = slot.nodes[2];
            const ggml_tensor * qg         = slot.nodes[3];
            const bool          bf16       = projection->src[0]->type == GGML_TYPE_BF16;
            result.executor                 = "ggml_backend_dayan_gated_output_projection";
            result.launch_name              = "gate-out";
            if (qg->type == GGML_TYPE_F32) {
                result.kernel = bf16 ? "project_gated_output_f32_gate_bf16_4c" :
                                       "project_gated_output_f32_gate_fp16_4c";
            } else {
                result.kernel = bf16 ? "project_gated_output_bf16_4c" : "project_gated_output_fp16_4c";
            }
        } break;
        case ggml_backend_dayan_slot_kind::ssm_conv:
        case ggml_backend_dayan_slot_kind::ssm_conv_silu:
            result = { "ggml_backend_dayan_ssm_conv", "ssm-conv", "ssm_conv_f32_1c", "single-core",
                       dayan_ssm_conv_task_count((uint32_t) node->ne[0], (uint32_t) node->ne[1], (uint32_t) node->ne[2]) };
            break;
    }
    const bool cooperative = topology.clusters && std::strcmp(result.task_mode, "four-core") == 0;
    result.task_mode = cooperative ? "four-core" : "single-core";
    result.task_dim_x = cooperative ? 4 : topology.cores;
    result.task_dim_y = cooperative ? topology.clusters : 1;
    if (!cooperative && result.kernel.size() >= 3 && result.kernel.compare(result.kernel.size() - 3, 3, "_4c") == 0) {
        result.kernel.replace(result.kernel.size() - 3, 3, "_1c");
    }
    if (slot.kind == ggml_backend_dayan_slot_kind::gated_delta_net) {
        result.task_dim_x = topology.launch(false, node->src[2]->ne[1] * node->src[2]->ne[3]).task_dim.x;
    }
    if (slot.kind == ggml_backend_dayan_slot_kind::ssm_conv || slot.kind == ggml_backend_dayan_slot_kind::ssm_conv_silu) {
        result.task_dim_x = dayan_ssm_conv_task_count(node->ne[0], node->ne[1], node->ne[2], topology.cores);
    }
    return result;
}

static void ggml_backend_dayan_plan_add_input(ggml_backend_dayan_graph_plan_slot * result,
                                               const ggml_tensor *                  tensor,
                                               const char *                         role) {
    if (tensor == nullptr) {
        return;
    }
    for (size_t i = 0; i < result->input_count; ++i) {
        if (result->inputs[i] == tensor) {
            return;
        }
    }
    GGML_ASSERT(result->input_count < GGML_DAYAN_GRAPH_PLAN_MAX_INPUTS);
    result->inputs[result->input_count]      = tensor;
    result->input_roles[result->input_count] = role;
    ++result->input_count;
}

static const ggml_tensor * ggml_backend_dayan_norm_scale(const ggml_tensor * mul, const ggml_tensor * rms);

static void ggml_backend_dayan_slot_inputs(const ggml_backend_dayan_graph_slot & slot,
                                           ggml_backend_dayan_graph_plan_slot *   result) {
    ggml_tensor * node = slot.nodes[0];
    switch (slot.kind) {
        case ggml_backend_dayan_slot_kind::mul_mat:
        case ggml_backend_dayan_slot_kind::mul_mat_silu:
        case ggml_backend_dayan_slot_kind::mul_mat_quantized:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "weight");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "activation");
            break;
        case ggml_backend_dayan_slot_kind::mul_mat_pair:
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[0], "weight-0");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[1]->src[0], "weight-1");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[1], "activation");
            break;
        case ggml_backend_dayan_slot_kind::mul_mat_swiglu:
        case ggml_backend_dayan_slot_kind::mul_mat_geglu:
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[0], "gate-weight");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[1]->src[0], "up-weight");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[1], "activation");
            break;
        case ggml_backend_dayan_slot_kind::mul_mat_id:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "weight");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "activation");
            ggml_backend_dayan_plan_add_input(result, node->src[2], "expert-ids");
            break;
        case ggml_backend_dayan_slot_kind::mul_mat_id_pair:
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[0], "weight-0");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[1]->src[0], "weight-1");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[1], "activation");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[2], "expert-ids");
            break;
        case ggml_backend_dayan_slot_kind::mul_mat_id_swiglu:
        case ggml_backend_dayan_slot_kind::mul_mat_id_geglu:
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[0], "gate-weight");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[1]->src[0], "up-weight");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[1], "activation");
            ggml_backend_dayan_plan_add_input(result, slot.nodes[0]->src[2], "expert-ids");
            break;
        case ggml_backend_dayan_slot_kind::add_id:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "input");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "bias");
            ggml_backend_dayan_plan_add_input(result, node->src[2], "expert-ids");
            break;
        case ggml_backend_dayan_slot_kind::elementwise_add:
        case ggml_backend_dayan_slot_kind::elementwise_mul:
        case ggml_backend_dayan_slot_kind::elementwise_sub:
        case ggml_backend_dayan_slot_kind::elementwise_div:
        case ggml_backend_dayan_slot_kind::concat:
        case ggml_backend_dayan_slot_kind::residual_add:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "left");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "right");
            break;
        case ggml_backend_dayan_slot_kind::get_rows:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "source");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "indices");
            break;
        case ggml_backend_dayan_slot_kind::set_rows:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "source");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "indices");
            ggml_backend_dayan_plan_add_input(result, node->src[2], "base");
            break;
        case ggml_backend_dayan_slot_kind::set:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "base");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "source");
            break;
        case ggml_backend_dayan_slot_kind::softmax:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "input");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "mask");
            ggml_backend_dayan_plan_add_input(result, node->src[2], "sinks");
            break;
        case ggml_backend_dayan_slot_kind::rope:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "input");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "positions");
            ggml_backend_dayan_plan_add_input(result, node->src[2], "factors");
            break;
        case ggml_backend_dayan_slot_kind::norm_pipeline: {
            ggml_tensor * add    = slot.nodes[0];
            ggml_tensor * rms    = slot.nodes[1];
            ggml_tensor * scaled = slot.nodes[2];
            if (add != nullptr) {
                ggml_backend_dayan_plan_add_input(result, add->src[0], "input");
                ggml_backend_dayan_plan_add_input(result, add->src[1], "residual");
            } else {
                ggml_backend_dayan_plan_add_input(result, rms->src[0], "input");
            }
            ggml_backend_dayan_plan_add_input(result, ggml_backend_dayan_norm_scale(scaled, rms), "scale");
        } break;
        case ggml_backend_dayan_slot_kind::flash_attention:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "query");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "key");
            ggml_backend_dayan_plan_add_input(result, node->src[2], "value");
            break;
        case ggml_backend_dayan_slot_kind::gated_delta_net:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "query");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "key");
            ggml_backend_dayan_plan_add_input(result, node->src[2], "value");
            ggml_backend_dayan_plan_add_input(result, node->src[3], "gate");
            ggml_backend_dayan_plan_add_input(result, node->src[4], "beta");
            ggml_backend_dayan_plan_add_input(result, node->src[5], "state");
            break;
        case ggml_backend_dayan_slot_kind::gated_output_projection: {
            ggml_tensor * sigmoid   = slot.nodes[0];
            ggml_tensor * gated     = slot.nodes[1];
            ggml_tensor * project   = slot.nodes[2];
            ggml_tensor * qg        = slot.nodes[3];
            ggml_tensor * attention = gated->src[0] == sigmoid ? gated->src[1] : gated->src[0];
            ggml_backend_dayan_plan_add_input(result, attention, "attention");
            ggml_backend_dayan_plan_add_input(result, qg, "gate");
            ggml_backend_dayan_plan_add_input(result, project->src[0], "weight");
        } break;
        case ggml_backend_dayan_slot_kind::ssm_conv:
        case ggml_backend_dayan_slot_kind::ssm_conv_silu:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "input");
            ggml_backend_dayan_plan_add_input(result, node->src[1], "weight");
            break;
        default:
            ggml_backend_dayan_plan_add_input(result, node->src[0], "input");
            break;
    }
}

bool ggml_backend_dayan_get_graph_plan(const ggml_cgraph *             graph,
                                       ggml_backend_dayan_graph_plan_callback callback,
                                       void *                                 user_data) {
    if (graph == nullptr || callback == nullptr) {
        return false;
    }

    DayanTopology topology;
    {
        std::lock_guard<std::mutex> lock(ggml_backend_dayan_device_mutex);
        if (!ggml_backend_dayan_device) { return false; }
        topology = ggml_backend_dayan_device->topology;
    }
    ggml_backend_dayan_graph_plan plan;
    const ggml_tensor *           unsupported = nullptr;
    if (!ggml_backend_dayan_compile_graph(graph, &plan, &unsupported)) {
        return false;
    }

    for (size_t slot_index = 0; slot_index < plan.slots.size(); ++slot_index) {
        const ggml_backend_dayan_graph_slot & slot = plan.slots[slot_index];
        const ggml_backend_dayan_slot_implementation implementation =
            ggml_backend_dayan_slot_implementation_info(slot, topology);
        ggml_backend_dayan_graph_plan_slot result = {};
        result.version                            = 3;
        result.kind                               = ggml_backend_dayan_slot_kind_name(slot.kind);
        result.executor                           = implementation.executor;
        result.launch_name                        = implementation.launch_name;
        result.kernel                             = implementation.kernel.c_str();
        result.task_mode                          = implementation.task_mode;
        result.task_dim_x                         = implementation.task_dim_x;
        result.task_dim_y                         = implementation.task_dim_y;
        result.task_dim_z                         = 1;
        result.slot_index                         = slot_index;
        result.node_count                         = slot.node_count;
        for (size_t i = 0; i < slot.nodes.size(); ++i) {
            result.nodes[i] = slot.nodes[i];
        }
        ggml_backend_dayan_slot_inputs(slot, &result);
        result.output_count = ggml_backend_dayan_slot_outputs(slot, result.outputs);
        result.output_mask  = slot.output_mask;
        result.sort_k       = slot.sort_k;
        callback(&result, user_data);
    }
    return true;
}

static ggml_backend_dayan_slot_trace_event ggml_backend_dayan_slot_event(const ggml_backend_dayan_graph_slot & slot,
                                                                         size_t                                slot_index) {
    ggml_backend_dayan_slot_trace_event event = {};
    event.version                             = 4;
    event.kind                                = ggml_backend_dayan_slot_kind_name(slot.kind);
    event.slot_index                          = slot_index;
    event.node_count                          = slot.node_count;
    for (size_t i = 0; i < slot.nodes.size(); ++i) {
        event.nodes[i] = slot.nodes[i];
    }
    event.output_count = ggml_backend_dayan_slot_outputs(slot, event.outputs);
    event.output_mask  = slot.output_mask;
    event.sort_k       = slot.sort_k;
    return event;
}

static bool ggml_backend_dayan_filter_slot(const ggml_backend_dayan_graph_slot & slot, size_t slot_index) {
    ggml_backend_dayan_slot_filter_callback callback  = nullptr;
    void *                                  user_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(ggml_backend_dayan_trace_mutex);
        callback  = ggml_backend_dayan_filter_callback;
        user_data = ggml_backend_dayan_filter_user_data;
    }
    if (callback == nullptr) {
        return true;
    }

    const ggml_backend_dayan_slot_trace_event event = ggml_backend_dayan_slot_event(slot, slot_index);
    return callback(&event, user_data);
}

static void ggml_backend_dayan_trace_slot(const ggml_backend_dayan_graph_slot & slot,
                                          size_t                                slot_index,
                                          bool                                  executed,
                                          int                                   status,
                                          int64_t                               elapsed_us) {
    ggml_backend_dayan_slot_trace_callback callback  = nullptr;
    void *                                 user_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(ggml_backend_dayan_trace_mutex);
        callback  = ggml_backend_dayan_trace_callback;
        user_data = ggml_backend_dayan_trace_user_data;
    }
    if (callback == nullptr) {
        return;
    }

    ggml_backend_dayan_slot_trace_event event = ggml_backend_dayan_slot_event(slot, slot_index);
    event.executed                            = executed;
    event.status                              = status;
    event.elapsed_us                          = elapsed_us;
    callback(&event, user_data);
}

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
            npu_runtime_topology_t topology = {};
            status = npu_runtime_get_topology(device->runtime, &topology);
            if (status == 0 && !device->topology.init(topology)) {
                status = -1;
            }
            if (status < 0) {
                GGML_LOG_ERROR("%s: cannot query Dayan topology (status %d); update Dayan Runtime and server\n", __func__, status);
            }
        }
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
    npu_buffer_t * config      = nullptr;
    npu_buffer_t * weight0     = nullptr;
    npu_buffer_t * weight1     = nullptr;
    npu_buffer_t * input2      = nullptr;
    npu_buffer_t * input3      = nullptr;
    npu_buffer_t * output      = nullptr;
    npu_buffer_t * output1     = nullptr;
    npu_buffer_t * result      = nullptr;
    bool           owns_config = true;
    bool           owns_input2 = true;
    bool           owns_result = true;
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
    size_t   runtime_bf16_casts        = 0;
    size_t   runtime_f16_casts         = 0;
    uint64_t runtime_bf16_cast_elements = 0;
    uint64_t runtime_f16_cast_elements  = 0;
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
    if (buffers->result != nullptr && buffers->owns_result) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->result), status);
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
    if (buffers->input2 != nullptr && buffers->owns_input2) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->input2), status);
    }
    if (buffers->weight1 != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->weight1), status);
    }
    if (buffers->weight0 != nullptr) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->weight0), status);
    }
    if (buffers->config != nullptr && buffers->owns_config) {
        ggml_backend_dayan_preserve_error(npu_buffer_destroy(buffers->config), status);
    }
}

static int ggml_backend_dayan_reserve_slot_buffer(npu_context_t * context,
                                                  size_t          size,
                                                  npu_buffer_t ** buffer,
                                                  size_t *        capacity) {
    if (*buffer != nullptr && *capacity >= size) {
        return 0;
    }

    npu_buffer_t * replacement = nullptr;
    int            status      = ggml_backend_dayan_buffer_create(context, size, &replacement);
    if (status < 0) {
        return status;
    }
    if (*buffer != nullptr) {
        status = npu_buffer_destroy(*buffer);
        if (status < 0) {
            npu_buffer_destroy(replacement);
            return status;
        }
    }
    *buffer   = replacement;
    *capacity = size;
    return 0;
}

static int ggml_backend_dayan_upload_slot_config(ggml_backend_dayan_context * ctx,
                                                 const void *                 config,
                                                 size_t                       config_size,
                                                 size_t                       config_capacity,
                                                 npu_buffer_t **              buffer) {
    if (ctx->active_slot >= ctx->slot_resources.size() || config == nullptr || config_size == 0) {
        return -1;
    }

    ggml_backend_dayan_slot_resources & resources = ctx->slot_resources[ctx->active_slot];
    if (resources.config_capacity < config_capacity) {
        resources.config_data.clear();
    }
    int status = ggml_backend_dayan_reserve_slot_buffer(ctx->context, config_capacity, &resources.config,
                                                        &resources.config_capacity);
    if (status < 0) {
        GGML_LOG_ERROR("%s: slot %zu failed to reserve %zu-byte config buffer with status %d\n", __func__,
                       ctx->active_slot, config_capacity, status);
    }
    if (status == 0 && (resources.config_data.size() != config_size ||
                        std::memcmp(resources.config_data.data(), config, config_size) != 0)) {
        status = ggml_backend_dayan_upload(ctx->stream, resources.config, config, config_size);
        if (status == 0) {
            resources.config_data.assign((const uint8_t *) config, (const uint8_t *) config + config_size);
        } else {
            GGML_LOG_ERROR("%s: slot %zu failed to upload %zu-byte config with status %d\n", __func__,
                           ctx->active_slot, config_size, status);
        }
    }
    if (status == 0) {
        *buffer = resources.config;
    }
    return status;
}

static int ggml_backend_dayan_prepare_launch_buffers_with_capacity(ggml_backend_dayan_context * ctx,
                                                                   const void *                 config,
                                                                   size_t                       config_size,
                                                                   size_t                       config_capacity,
                                                                   size_t                       result_size,
                                                                   ggml_backend_dayan_buffers * buffers) {
    npu_buffer_t * config_buffer = nullptr;
    int status = ggml_backend_dayan_upload_slot_config(ctx, config, config_size, config_capacity, &config_buffer);
    if (status < 0) {
        return status;
    }
    ggml_backend_dayan_slot_resources & resources = ctx->slot_resources[ctx->active_slot];
    if (result_size > 0) {
        status =
            ggml_backend_dayan_reserve_slot_buffer(ctx->context, result_size, &resources.result, &resources.result_capacity);
    }
    if (status == 0) {
        buffers->config      = config_buffer;
        buffers->result      = result_size > 0 ? resources.result : nullptr;
        buffers->owns_config = false;
        buffers->owns_result = false;
    }
    return status;
}

static int ggml_backend_dayan_prepare_launch_buffers(ggml_backend_dayan_context * ctx,
                                                     const void *                 config,
                                                     size_t                       config_size,
                                                     size_t                       result_size,
                                                     ggml_backend_dayan_buffers * buffers) {
    return ggml_backend_dayan_prepare_launch_buffers_with_capacity(ctx, config, config_size, config_size, result_size,
                                                                   buffers);
}

static int ggml_backend_dayan_prepare_slot_workspace(ggml_backend_dayan_context * ctx,
                                                     size_t                       size,
                                                     ggml_backend_dayan_buffers * buffers) {
    if (ctx->active_slot >= ctx->slot_resources.size() || size == 0) {
        return -1;
    }
    ggml_backend_dayan_slot_resources & resources = ctx->slot_resources[ctx->active_slot];
    const int                           status =
        ggml_backend_dayan_reserve_slot_buffer(ctx->context, size, &resources.workspace, &resources.workspace_capacity);
    if (status == 0) {
        buffers->input2      = resources.workspace;
        buffers->owns_input2 = false;
    }
    return status;
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
    if (tensor != nullptr && tensor->type == GGML_TYPE_MXFP4 && tensor->extra != nullptr) {
        return ((ggml_backend_dayan_mxfp4_extra *) tensor->extra)->data;
    }
    return tensor != nullptr ? (npu_buffer_t *) tensor->extra : nullptr;
}

static void ggml_backend_dayan_init_launch_attrs(ggml_backend_dayan_context * ctx,
                                                 const char *                 name,
                                                 npu_kernel_launch_attrs_t *  attrs);
static void ggml_backend_dayan_init_native_launch_attrs(ggml_backend_dayan_context * ctx,
                                                        const char *                 name,
                                                        uint64_t                     work_items,
                                                        npu_kernel_launch_attrs_t *  attrs);

static uint64_t ggml_backend_dayan_native_tiles(uint64_t elements) {
    return elements == 0 ? 0 : 1 + (elements - 1) / kDayanNativeTaskTileElements;
}

static uint64_t ggml_backend_dayan_copy_work_items(const DayanCopyConfig & config) {
    if (config.source_type == config.destination_type) {
        return 1;
    }
    const uint64_t rows = config.elements / config.source_ne[0];
    return rows * ggml_backend_dayan_native_tiles(config.source_ne[0]);
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
        ggml_backend_dayan_buffers launch_buffers;
        status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &launch_buffers);
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            ggml_backend_dayan_init_native_launch_attrs(
                ctx, "activation-cast", ggml_backend_dayan_copy_work_items(config), &attrs);
            status =
                copy_f32_1c(launch_buffers.config, ggml_backend_dayan_tensor_buffer(activation), device_buffer, &attrs);
        }
        ggml_backend_dayan_buffers_free(&launch_buffers, &status);
    } else if (status == 0) {
        status = -1;
    }

    if (status == 0) {
        buffers->activation_by_tensor.emplace(key, device_buffer);
        if (storage_type == GGML_TYPE_BF16) {
            ++buffers->runtime_bf16_casts;
            buffers->runtime_bf16_cast_elements += activation_count;
        } else {
            ++buffers->runtime_f16_casts;
            buffers->runtime_f16_cast_elements += activation_count;
        }
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
                                                        uint64_t                     work_items,
                                                        npu_kernel_launch_attrs_t *  attrs) {
    ggml_backend_dayan_init_launch_attrs(ctx, name, attrs);
    GGML_UNUSED(work_items);
    attrs->config = ctx->device->topology.launch(false);
}

static void ggml_backend_dayan_init_matmul_launch_attrs(ggml_backend_dayan_context * ctx,
                                                        const char *                 name,
                                                        npu_kernel_launch_attrs_t *  attrs) {
    ggml_backend_dayan_init_launch_attrs(ctx, name, attrs);
    attrs->config = ctx->device->topology.launch(true);
}

template <typename Kernel>
static Kernel ggml_backend_dayan_matmul_kernel(ggml_backend_dayan_context * ctx, Kernel single_core, Kernel four_core) {
    return ctx->device->topology.clusters ? four_core : single_core;
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

static int ggml_backend_dayan_quantize_mxfp4(ggml_backend_dayan_context * ctx, const ggml_tensor * input,
                                           npu_buffer_t * output, npu_buffer_t * scales) {
    const auto layout = DayanMxfp4Layout::make(ggml_nrows(input), input->ne[0]);
    DayanMxfp4Config config = {};
    config.m = layout.rows;
    config.n = layout.columns;
    config.a_stride = input->nb[1];
    config.output_stride = layout.row_stride;
    config.output_scale_stride = layout.scale_stride;
    if (!ggml_backend_dayan_type_code(input->type, &config.source_type)) { return -1; }
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "quantize-mxfp4", layout.rows, &attrs);
        status = quantize_mxfp4(buffers.config, ggml_backend_dayan_tensor_buffer(input), output, scales, &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static int ggml_backend_dayan_mxfp4_matmul(ggml_backend_dayan_context * ctx, ggml_backend_dayan_graph_buffers * graph,
                                          const ggml_tensor * node, const ggml_tensor * up, ggml_tensor * output,
                                          uint32_t operation) {
    const ggml_tensor * weight = node->src[0], * input = node->src[1];
    const bool mx = weight->type == GGML_TYPE_MXFP4;
    const auto a_layout = DayanMxfp4Layout::make(ggml_nrows(input), input->ne[0]);
    npu_buffer_t * a = nullptr, * as = nullptr;
    int status = 0;
    if (mx) {
        if (input->type == GGML_TYPE_MXFP4) {
            a = ggml_backend_dayan_tensor_buffer(input);
        } else {
            const ggml_backend_dayan_activation_key key = {input, GGML_TYPE_MXFP4};
            const auto cached = graph->activation_by_tensor.find(key);
            if (cached != graph->activation_by_tensor.end()) {
                a = cached->second;
            } else {
                status = ggml_backend_dayan_buffer_create(ctx->context, a_layout.bytes, &a);
                if (status < 0) { return status; }
                graph->activations.push_back(a);
                status = npu_buffer_create_view(a, a_layout.scale_offset, a_layout.bytes - a_layout.scale_offset, &as);
                if (status < 0) { return status; }
                status = ggml_backend_dayan_quantize_mxfp4(ctx, input, a, as);
                npu_buffer_destroy(as);
                as = nullptr;
                if (status < 0) { return status; }
                graph->activation_by_tensor.emplace(key, a);
            }
        }
        status = npu_buffer_create_view(a, a_layout.scale_offset, a_layout.bytes - a_layout.scale_offset, &as);
    } else {
        status = ggml_backend_dayan_prepare_activation(ctx, graph, input, weight->type, &a);
    }
    if (status < 0) { return status; }
    const auto * b = mx ? (const ggml_backend_dayan_mxfp4_extra *) weight->extra : nullptr;
    const auto * u = mx && up != nullptr ? (const ggml_backend_dayan_mxfp4_extra *) up->src[0]->extra : b;
    const auto * out = output->type == GGML_TYPE_MXFP4 ? (const ggml_backend_dayan_mxfp4_extra *) output->extra : nullptr;
    DayanMxfp4Config config = {};
    config.m = ggml_nrows(input); config.n = weight->ne[1]; config.k = weight->ne[0];
    config.a_stride = mx ? a_layout.row_stride : config.k * 2;
    config.b_stride = mx ? b->layout.row_stride : weight->nb[1];
    config.output_stride = out ? out->layout.row_stride : output->nb[1];
    config.a_scale_stride = a_layout.scale_stride;
    config.b_scale_stride = mx ? b->layout.scale_stride : 0;
    config.output_scale_stride = out ? out->layout.scale_stride : 0;
    config.weight_type = mx ? kDayanDataTypeMXFP4 : weight->type == GGML_TYPE_BF16 ? kDayanDataTypeBF16 : kDayanDataTypeF16;
    config.source_type = config.weight_type;
    config.output_type = out ? kDayanDataTypeMXFP4 : output->type == GGML_TYPE_F32 ? kDayanDataTypeF32 :
                         output->type == GGML_TYPE_BF16 ? kDayanDataTypeBF16 : kDayanDataTypeF16;
    config.operation = operation;
    ggml_backend_dayan_buffers buffers;
    status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, "matmul-mxfp4", &attrs);
        npu_buffer_t * weight_buffer = ggml_backend_dayan_tensor_buffer(weight);
        status = ggml_backend_dayan_matmul_kernel(ctx, matmul_mxfp4_1c, matmul_mxfp4_4c)(buffers.config, a, as ? as : a, weight_buffer, mx ? b->scales : weight_buffer,
                                up ? ggml_backend_dayan_tensor_buffer(up->src[0]) : weight_buffer,
                                mx ? u->scales : weight_buffer, ggml_backend_dayan_tensor_buffer(output),
                                out ? out->scales : ggml_backend_dayan_tensor_buffer(output), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (as != nullptr) { ggml_backend_dayan_preserve_error(npu_buffer_destroy(as), &status); }
    return status;
}

static bool ggml_backend_dayan_is_mxfp4_slot(const ggml_backend_dayan_graph_slot & slot) {
    switch (slot.kind) {
        case ggml_backend_dayan_slot_kind::copy:
            return slot.nodes[0]->type == GGML_TYPE_MXFP4;
        case ggml_backend_dayan_slot_kind::mul_mat_quantized:
            return slot.nodes[0]->src[0]->type == GGML_TYPE_MXFP4 || slot.nodes[1]->type == GGML_TYPE_MXFP4;
        case ggml_backend_dayan_slot_kind::mul_mat_swiglu:
        case ggml_backend_dayan_slot_kind::mul_mat_geglu:
            return slot.nodes[0]->src[0]->type == GGML_TYPE_MXFP4 || slot.nodes[3]->type == GGML_TYPE_MXFP4;
        case ggml_backend_dayan_slot_kind::mul_mat:
        case ggml_backend_dayan_slot_kind::mul_mat_silu:
        case ggml_backend_dayan_slot_kind::mul_mat_pair:
            return slot.nodes[0]->src[0]->type == GGML_TYPE_MXFP4;
        default: return false;
    }
}

static int ggml_backend_dayan_mxfp4_slot(ggml_backend_dayan_context * ctx, ggml_backend_dayan_graph_buffers * graph,
                                        const ggml_backend_dayan_graph_slot & slot) {
    if (slot.kind == ggml_backend_dayan_slot_kind::copy) {
        const auto * extra = (const ggml_backend_dayan_mxfp4_extra *) slot.nodes[0]->extra;
        return ggml_backend_dayan_quantize_mxfp4(ctx, slot.nodes[0]->src[0], extra->data, extra->scales);
    }
    if (slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu || slot.kind == ggml_backend_dayan_slot_kind::mul_mat_geglu) {
        return ggml_backend_dayan_mxfp4_matmul(ctx, graph, slot.nodes[0], slot.nodes[1], slot.nodes[3],
                                              slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ? 2 : 3);
    }
    const bool silu = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_silu;
    ggml_tensor * output = silu || slot.kind == ggml_backend_dayan_slot_kind::mul_mat_quantized ? slot.nodes[1] : slot.nodes[0];
    int status = ggml_backend_dayan_mxfp4_matmul(ctx, graph, slot.nodes[0], nullptr, output, silu ? 1 : 0);
    if (status == 0 && slot.kind == ggml_backend_dayan_slot_kind::mul_mat_pair) {
        status = ggml_backend_dayan_mxfp4_matmul(ctx, graph, slot.nodes[1], nullptr, slot.nodes[1], 0);
    }
    return status;
}

static DayanMatmulConfig ggml_backend_dayan_matmul_config(const ggml_tensor * weight,
                                                          const ggml_tensor * activation,
                                                          const ggml_tensor * output) {
    DayanMatmulConfig config = {};
    config.m                 = activation->ne[1];
    config.n                 = weight->ne[1];
    config.k                 = weight->ne[0];
    config.a_stride          = config.k * sizeof(uint16_t);
    config.bt_stride         = weight->nb[1];
    config.output_stride     = output->nb[1];

    if (weight->ne[2] == 1 && weight->ne[3] == 1) {
        config.m *= (uint64_t) activation->ne[2] * (uint64_t) activation->ne[3];
        return config;
    }

    config.batch_ne2            = activation->ne[2];
    config.batch_ne3            = activation->ne[3];
    config.weight_ne2           = weight->ne[2];
    config.weight_ne3           = weight->ne[3];
    config.a_batch_stride2      = config.a_stride * (uint64_t) activation->ne[1];
    config.a_batch_stride3      = config.a_batch_stride2 * config.batch_ne2;
    config.bt_batch_stride2     = weight->nb[2];
    config.bt_batch_stride3     = weight->nb[3];
    config.output_batch_stride2 = output->nb[2];
    config.output_batch_stride3 = output->nb[3];
    return config;
}

static int ggml_backend_dayan_mul_mat(ggml_backend_dayan_context * ctx,
                                      const ggml_tensor *          mul_mat,
                                      ggml_tensor *                output,
                                      npu_buffer_t *               activation_buffer,
                                      bool                         silu) {
    const ggml_tensor * weight     = mul_mat->src[0];
    const ggml_tensor * activation = mul_mat->src[1];
    const DayanMatmulConfig config = ggml_backend_dayan_matmul_config(weight, activation, output);
    ggml_backend_dayan_buffers buffers;
    int                        status =
        ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, silu ? "ggml-mm-silu" : "ggml-mul-mat", &attrs);
        if (weight->type == GGML_TYPE_BF16) {
            status = silu ?
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_silu_inference_bf16_1c, matmul_silu_inference_bf16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_silu_bf16_1c, matmul_silu_bf16_4c))(
                             buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                             ggml_backend_dayan_tensor_buffer(output), &attrs) :
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_inference_bf16_1c, matmul_inference_bf16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_bf16_1c, matmul_bf16_4c))(
                             buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                             ggml_backend_dayan_tensor_buffer(output), &attrs);
        } else {
            status = silu ?
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_silu_inference_fp16_1c, matmul_silu_inference_fp16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_silu_fp16_1c, matmul_silu_fp16_4c))(
                             buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                             ggml_backend_dayan_tensor_buffer(output), &attrs) :
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_inference_fp16_1c, matmul_inference_fp16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_fp16_1c, matmul_fp16_4c))(
                             buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                             ggml_backend_dayan_tensor_buffer(output), &attrs);
        }
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan%s matrix multiplication failed with status %d\n", __func__,
                       silu ? " SiLU" : "", status);
    }
    return status;
}

static int ggml_backend_dayan_mul_mat_pair(ggml_backend_dayan_context *          ctx,
                                           const ggml_backend_dayan_graph_slot & slot,
                                           npu_buffer_t *                        activation_buffer) {
    ggml_tensor *       first             = slot.nodes[0];
    ggml_tensor *       second            = slot.nodes[1];
    const ggml_tensor * first_weight      = first->src[0];
    const ggml_tensor * second_weight     = second->src[0];
    const ggml_tensor * activation        = first->src[1];
    const DayanMatmulPairConfig config = {
        { (uint64_t) activation->ne[1], (uint64_t) first_weight->ne[1], (uint64_t) first_weight->ne[0],
          (uint64_t) first_weight->ne[0] * sizeof(uint16_t), first_weight->nb[1], first->nb[1] },
        { (uint64_t) activation->ne[1], (uint64_t) second_weight->ne[1], (uint64_t) second_weight->ne[0],
          (uint64_t) second_weight->ne[0] * sizeof(uint16_t), second_weight->nb[1], second->nb[1] },
    };
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(
        ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, "ggml-mm-pair", &attrs);
        status = first_weight->type == GGML_TYPE_BF16 ?
                     ggml_backend_dayan_matmul_kernel(ctx, matmul_pair_bf16_1c, matmul_pair_bf16_4c)(
                         buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(first_weight),
                         ggml_backend_dayan_tensor_buffer(second_weight), ggml_backend_dayan_tensor_buffer(first),
                         ggml_backend_dayan_tensor_buffer(second), &attrs) :
                     ggml_backend_dayan_matmul_kernel(ctx, matmul_pair_fp16_1c, matmul_pair_fp16_4c)(
                         buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(first_weight),
                         ggml_backend_dayan_tensor_buffer(second_weight), ggml_backend_dayan_tensor_buffer(first),
                         ggml_backend_dayan_tensor_buffer(second), &attrs);
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan paired matrix multiplication failed with status %d\n", __func__, status);
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
    int                        status =
        ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, "ggml-mm-lowp", &attrs);
        status =
            weight->type == GGML_TYPE_BF16 ?
                (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_quantized_inference_bf16_1c, matmul_quantized_inference_bf16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_quantized_bf16_1c, matmul_quantized_bf16_4c))(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                         ggml_backend_dayan_tensor_buffer(output), &attrs) :
                (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_quantized_inference_fp16_1c, matmul_quantized_inference_fp16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_quantized_fp16_1c, matmul_quantized_fp16_4c))(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                         ggml_backend_dayan_tensor_buffer(output), &attrs);
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
    int                        status =
        ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(
            ctx, slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ? "ggml-swiglu" : "ggml-geglu", &attrs);

        if (gate->src[0]->type == GGML_TYPE_BF16) {
            status = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ?
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_swiglu_inference_bf16_1c, matmul_swiglu_inference_bf16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_swiglu_bf16_1c, matmul_swiglu_bf16_4c))(buffers.config, activation_buffer,
                                               ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                               ggml_backend_dayan_tensor_buffer(up->src[0]),
                                               ggml_backend_dayan_tensor_buffer(output), &attrs) :
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_geglu_inference_bf16_1c, matmul_geglu_inference_bf16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_geglu_bf16_1c, matmul_geglu_bf16_4c))(buffers.config, activation_buffer,
                                              ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                              ggml_backend_dayan_tensor_buffer(up->src[0]),
                                              ggml_backend_dayan_tensor_buffer(output), &attrs);
        } else {
            status = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_swiglu ?
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_swiglu_inference_fp16_1c, matmul_swiglu_inference_fp16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_swiglu_fp16_1c, matmul_swiglu_fp16_4c))(buffers.config, activation_buffer,
                                               ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                               ggml_backend_dayan_tensor_buffer(up->src[0]),
                                               ggml_backend_dayan_tensor_buffer(output), &attrs) :
                         (config.m <= 512 ? ggml_backend_dayan_matmul_kernel(ctx, matmul_geglu_inference_fp16_1c, matmul_geglu_inference_fp16_4c) : ggml_backend_dayan_matmul_kernel(ctx, matmul_geglu_fp16_1c, matmul_geglu_fp16_4c))(buffers.config, activation_buffer,
                                              ggml_backend_dayan_tensor_buffer(gate->src[0]),
                                              ggml_backend_dayan_tensor_buffer(up->src[0]),
                                              ggml_backend_dayan_tensor_buffer(output), &attrs);
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
        dst->src[2]->nb[1],
    };
}

static int ggml_backend_dayan_mul_mat_id(ggml_backend_dayan_context * ctx,
                                         ggml_tensor *                dst,
                                         npu_buffer_t *               activation_buffer) {
    const ggml_tensor *        weight = dst->src[0];
    const DayanMulMatIdConfig  config = ggml_backend_dayan_mul_mat_id_config(dst);
    ggml_backend_dayan_buffers buffers;
    int                        status =
        ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, "ggml-mul-mat-id", &attrs);
        status = weight->type == GGML_TYPE_BF16 ?
                     ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_bf16_1c, mul_mat_id_bf16_4c)(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                        ggml_backend_dayan_tensor_buffer(dst->src[2]),
                                        ggml_backend_dayan_tensor_buffer(dst), &attrs) :
                     ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_fp16_1c, mul_mat_id_fp16_4c)(buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(weight),
                                        ggml_backend_dayan_tensor_buffer(dst->src[2]),
                                        ggml_backend_dayan_tensor_buffer(dst), &attrs);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan expert matrix multiplication failed with status %d\n", __func__, status);
    }
    return status;
}

static int ggml_backend_dayan_mul_mat_id_pair(ggml_backend_dayan_context *          ctx,
                                              const ggml_backend_dayan_graph_slot & slot,
                                              npu_buffer_t *                        activation_buffer) {
    ggml_tensor *       first         = slot.nodes[0];
    ggml_tensor *       second        = slot.nodes[1];
    const ggml_tensor * first_weight  = first->src[0];
    const ggml_tensor * second_weight = second->src[0];
    const DayanMulMatIdPairConfig config = {
        ggml_backend_dayan_mul_mat_id_config(first),
        ggml_backend_dayan_mul_mat_id_config(second),
    };
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(
        ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, "ggml-moe-pair", &attrs);
        status = first_weight->type == GGML_TYPE_BF16 ?
                     ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_pair_bf16_1c, mul_mat_id_pair_bf16_4c)(
                         buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(first_weight),
                         ggml_backend_dayan_tensor_buffer(second_weight),
                         ggml_backend_dayan_tensor_buffer(first->src[2]), ggml_backend_dayan_tensor_buffer(first),
                         ggml_backend_dayan_tensor_buffer(second), &attrs) :
                     ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_pair_fp16_1c, mul_mat_id_pair_fp16_4c)(
                         buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(first_weight),
                         ggml_backend_dayan_tensor_buffer(second_weight),
                         ggml_backend_dayan_tensor_buffer(first->src[2]), ggml_backend_dayan_tensor_buffer(first),
                         ggml_backend_dayan_tensor_buffer(second), &attrs);
    }

    ggml_backend_dayan_buffers_free(&buffers, &status);
    if (status < 0) {
        GGML_LOG_ERROR("%s: Dayan paired expert matrix multiplication failed with status %d\n", __func__, status);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        const bool                swiglu = slot.kind == ggml_backend_dayan_slot_kind::mul_mat_id_swiglu;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, swiglu ? "ggml-moe-swiglu" : "ggml-moe-geglu", &attrs);
        if (gate->src[0]->type == GGML_TYPE_BF16) {
            status =
                swiglu ?
                    ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_swiglu_bf16_1c, mul_mat_id_swiglu_bf16_4c)(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), &attrs) :
                    ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_geglu_bf16_1c, mul_mat_id_geglu_bf16_4c)(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), &attrs);
        } else {
            status =
                swiglu ?
                    ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_swiglu_fp16_1c, mul_mat_id_swiglu_fp16_4c)(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), &attrs) :
                    ggml_backend_dayan_matmul_kernel(ctx, mul_mat_id_geglu_fp16_1c, mul_mat_id_geglu_fp16_4c)(
                        buffers.config, activation_buffer, ggml_backend_dayan_tensor_buffer(gate->src[0]),
                        ggml_backend_dayan_tensor_buffer(up->src[0]), ggml_backend_dayan_tensor_buffer(gate->src[2]),
                        ggml_backend_dayan_tensor_buffer(output), &attrs);
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
        output->src[2]->nb[1],
    };
    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "ggml-add-id", (uint64_t) config.tokens * config.experts_used, &attrs);
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
        slot.sort_k,
        top_k || ggml_get_op_params_i32(output, 0) == GGML_SORT_ORDER_DESC ? kDayanSortOrderDesc : kDayanSortOrderAsc,
        (uint32_t) output->ne[0],
    };

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, top_k ? "ggml-top-k" : slot.sort_k < output->ne[0] ? "ggml-sort-topk" : "ggml-argsort", config.rows, &attrs);
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
    const float inv_width      = 1.0f / (float) config.width;
    std::memcpy(&config.inv_width_bits, &inv_width, sizeof(config.inv_width_bits));
    GGML_ASSERT(ggml_backend_dayan_type_code(input->type, &config.input_type));
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
    const size_t config_capacity = std::max({ sizeof(config), sizeof(DayanRepeatConfig), sizeof(DayanCopyConfig) });
    int                          status = ggml_backend_dayan_prepare_launch_buffers_with_capacity(
        ctx, &config, sizeof(config), config_capacity, 0, &buffers);
    npu_buffer_t * inputs[] = { ggml_backend_dayan_tensor_buffer(left), ggml_backend_dayan_tensor_buffer(right) };
    npu_buffer_t ** packed[] = { &buffers.weight0, &buffers.input3 };
    for (int source = 0; source < 2 && status == 0; ++source) {
        const ggml_tensor * input = output->src[source];
        if (ggml_is_contiguous(input)) {
            continue;
        }
        DayanCopyConfig copy = {};
        copy.elements = (uint64_t) ggml_nelements(input);
        copy.source_type = copy.destination_type = kDayanDataTypeF32;
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            copy.source_ne[dimension] = (uint32_t) input->ne[dimension];
            copy.source_nb[dimension] = input->nb[dimension];
        }
        status = ggml_backend_dayan_buffer_create(ctx->context, copy.elements * sizeof(float), packed[source]);
        if (status == 0) {
            status = ggml_backend_dayan_upload_slot_config(ctx, &copy, sizeof(copy), config_capacity, &buffers.config);
        }
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            ggml_backend_dayan_init_native_launch_attrs(ctx, "binary-pack", ggml_backend_dayan_copy_work_items(copy), &attrs);
            status = copy_f32_1c(buffers.config, inputs[source], *packed[source], &attrs);
            inputs[source] = *packed[source];
        }
    }
    npu_buffer_t * right_buffer = inputs[1];
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
            status =
                ggml_backend_dayan_upload_slot_config(ctx, &repeat, sizeof(repeat), config_capacity, &buffers.config);
        }
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            ggml_backend_dayan_init_native_launch_attrs(ctx, "repeat", 1, &attrs);
            status = repeat_1c(buffers.config, inputs[1], right_buffer, &attrs);
            if (status < 0) {
                GGML_LOG_ERROR("%s: Dayan elementwise repeat failed with status %d\n", __func__, status);
            }
        }
    }
    if (status == 0) {
        status = ggml_backend_dayan_upload_slot_config(ctx, &config, sizeof(config), config_capacity,
                                                       &buffers.config);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::elementwise_add:
                ggml_backend_dayan_init_native_launch_attrs(
                    ctx, "elementwise-add", ggml_backend_dayan_native_tiles(config.elements), &attrs);
                status = elementwise_add_f32_1c(buffers.config, inputs[0], right_buffer,
                                                ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_mul:
                ggml_backend_dayan_init_native_launch_attrs(
                    ctx, "elementwise-mul", ggml_backend_dayan_native_tiles(config.elements), &attrs);
                status = elementwise_mul_f32_1c(buffers.config, inputs[0], right_buffer,
                                                ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_sub:
                ggml_backend_dayan_init_native_launch_attrs(
                    ctx, "elementwise-sub", ggml_backend_dayan_native_tiles(config.elements), &attrs);
                status = elementwise_sub_f32_1c(buffers.config, inputs[0], right_buffer,
                                                ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::elementwise_div:
                ggml_backend_dayan_init_native_launch_attrs(
                    ctx, "elementwise-div", ggml_backend_dayan_native_tiles(config.elements), &attrs);
                status = elementwise_div_f32_1c(buffers.config, inputs[0], right_buffer,
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "scale", ggml_backend_dayan_native_tiles(config.elements), &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
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
            ggml_backend_dayan_init_native_launch_attrs(
                ctx, launch_name, ggml_backend_dayan_native_tiles(config.elements), &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::sum:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "sum", config.rows, &attrs);
                status = sum_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                    ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::sum_rows:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "sum-rows", config.rows, &attrs);
                status = sum_rows_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                         ggml_backend_dayan_tensor_buffer(output), &attrs);
                break;
            case ggml_backend_dayan_slot_kind::mean:
                ggml_backend_dayan_init_native_launch_attrs(ctx, "mean", config.rows, &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "l2-norm", config.rows, &attrs);
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

    uint32_t source_type = kDayanDataTypeMXFP4;
    if (source->type != GGML_TYPE_MXFP4 && !ggml_backend_dayan_type_code(source->type, &source_type)) {
        return -1;
    }

    DayanRowsConfig config  = {};
    config.row_elements     = (uint32_t) output->ne[0];
    config.rows             = (uint32_t) ggml_nrows(output);
    config.source_rows      = (uint64_t) ggml_nrows(source);
    config.destination_rows = (uint64_t) ggml_nrows(output);
    config.source_type      = source_type;
    if (source->type == GGML_TYPE_MXFP4) {
        const auto * extra = (const ggml_backend_dayan_mxfp4_extra *) source->extra;
        if (extra == nullptr) { return -1; }
        config.source_row_stride = extra->layout.row_stride;
        config.source_scale_stride = extra->layout.scale_stride;
        config.source_scale_offset = extra->layout.scale_offset;
    }
    if (!ggml_backend_dayan_type_code(output->type, &config.destination_type)) {
        return -1;
    }
    config.index_type       = indices->type == GGML_TYPE_I32 ? kDayanDataTypeI32 : kDayanDataTypeI64;
    for (int dimension = 0; dimension < 3; ++dimension) {
        config.index_ne[dimension]       = (uint32_t) indices->ne[dimension];
        config.index_nb[dimension]       = indices->nb[dimension];
        config.source_ne[dimension]      = (uint32_t) source->ne[dimension + 1];
        config.destination_ne[dimension] = (uint32_t) output->ne[dimension + 1];
    }

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "get-rows", (uint64_t) config.rows * ggml_backend_dayan_native_tiles(config.row_elements), &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        status = ggml_backend_dayan_copy(ctx->stream, ggml_backend_dayan_tensor_buffer(output), 0,
                                         ggml_backend_dayan_tensor_buffer(output->src[2]), 0, ggml_nbytes(output));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "set-rows", (uint64_t) config.rows * ggml_backend_dayan_native_tiles(config.row_elements), &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status < 0) {
        GGML_LOG_ERROR("%s: slot %zu failed before launch: %s %s -> %s, ne=[%" PRId64 ", %" PRId64 ", %" PRId64
                       ", %" PRId64 "], nb=[%zu, %zu, %zu, %zu], status=%d\n",
                       __func__, ctx->active_slot, source->name, ggml_type_name(source->type),
                       ggml_type_name(output->type), source->ne[0], source->ne[1], source->ne[2], source->ne[3],
                       source->nb[0], source->nb[1], source->nb[2], source->nb[3], status);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "copy", ggml_backend_dayan_copy_work_items(config), &attrs);
        status = copy_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(source),
                             ggml_backend_dayan_tensor_buffer(output), &attrs);
        if (status < 0) {
            GGML_LOG_ERROR("%s: slot %zu launch failed: %s %s -> %s, ne=[%" PRId64 ", %" PRId64 ", %" PRId64
                           ", %" PRId64 "], nb=[%zu, %zu, %zu, %zu], status=%d\n",
                           __func__, ctx->active_slot, source->name, ggml_type_name(source->type),
                           ggml_type_name(output->type), source->ne[0], source->ne[1], source->ne[2], source->ne[3],
                           source->nb[0], source->nb[1], source->nb[2], source->nb[3], status);
        }
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "repeat", 1, &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        status = ggml_backend_dayan_copy(ctx->stream, ggml_backend_dayan_tensor_buffer(output), 0,
                                         ggml_backend_dayan_tensor_buffer(base), 0, ggml_nbytes(output));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "set", 1, &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
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
            const uint64_t work_items = slot.kind == ggml_backend_dayan_slot_kind::diag_mask_inf ||
                                                slot.kind == ggml_backend_dayan_slot_kind::diag_mask_zero ?
                                            config.output_ne[1] :
                                            1;
            ggml_backend_dayan_init_native_launch_attrs(ctx, launch_name, work_items, &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "clamp", ggml_backend_dayan_native_tiles(config.elements), &attrs);
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
    int            status      = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    npu_buffer_t * mask_buffer = ggml_backend_dayan_tensor_buffer(mask);
    npu_buffer_t * sink_buffer = ggml_backend_dayan_tensor_buffer(sinks);
    if (status == 0 && mask_buffer == nullptr) {
        status      = ggml_backend_dayan_buffer_create(ctx->context, sizeof(float), &buffers.weight1);
        mask_buffer = buffers.weight1;
    }
    if (status == 0 && sink_buffer == nullptr) {
        status      = ggml_backend_dayan_buffer_create(ctx->context, sizeof(float), &buffers.input2);
        sink_buffer = buffers.input2;
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "softmax", (uint64_t) config.batches * config.heads * config.rows, &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, "concat", 1, &attrs);
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

    DayanRopeConfig config = {};
    config.width           = (uint32_t) output->ne[0];
    config.heads           = (uint32_t) output->ne[1];
    config.tokens          = (uint32_t) output->ne[2];
    config.batches         = (uint32_t) output->ne[3];
    config.n_dims          = (uint32_t) n_dims;
    config.mode            = (uint32_t) mode;
    std::memcpy(&config.theta_scale_bits, &theta_scale, sizeof(theta_scale));
    std::memcpy(&config.freq_scale_bits, &freq_scale, sizeof(freq_scale));
    std::memcpy(&config.ext_factor_bits, &ext_factor, sizeof(ext_factor));
    std::memcpy(&config.mscale_bits, &mscale, sizeof(mscale));
    std::memcpy(&config.corr_dim0_bits, &corr_dims[0], sizeof(corr_dims[0]));
    std::memcpy(&config.corr_dim1_bits, &corr_dims[1], sizeof(corr_dims[1]));
    std::memcpy(config.sections, (const int32_t *) output->op_params + 11, sizeof(config.sections));
    config.axis_values[0] = 0;
    config.axis_values[1] = 1;
    config.axis_values[2] = 2;

    float theta_factor = 1.0f;
    for (int32_t pair = 0; pair < n_dims / 2; ++pair) {
        uint32_t axis = 0;
        if ((mode & GGML_ROPE_TYPE_MROPE) != 0) {
            const int32_t section_count =
                config.sections[0] + config.sections[1] + config.sections[2] + config.sections[3];
            if (section_count > 0) {
                const int32_t sector = pair % section_count;
                if (mode == GGML_ROPE_TYPE_IMROPE) {
                    if (sector % 3 == 1 && sector < 3 * config.sections[1]) {
                        axis = 1;
                    } else if (sector % 3 == 2 && sector < 3 * config.sections[2]) {
                        axis = 2;
                    } else if (sector % 3 != 0 || sector >= 3 * config.sections[0]) {
                        axis = 3;
                    }
                } else if (sector >= config.sections[0] + config.sections[1] + config.sections[2]) {
                    axis = 3;
                } else if (sector >= config.sections[0] + config.sections[1]) {
                    axis = 2;
                } else if (sector >= config.sections[0]) {
                    axis = 1;
                }
            }
        }
        config.axes[pair] = axis;

        const float range = std::max(corr_dims[1] - corr_dims[0], 0.001f);
        const float ramp  = std::clamp(((float) pair - corr_dims[0]) / range, 0.0f, 1.0f);
        const float mix   = (1.0f - ramp) * ext_factor;
        const float multiplier = theta_factor * (freq_scale * (1.0f - mix) + mix);
        std::memcpy(&config.multipliers[pair], &multiplier, sizeof(multiplier));
        theta_factor *= theta_scale;
    }

    ggml_backend_dayan_buffers buffers;
    int                status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    npu_buffer_t *     factor_buffer = ggml_backend_dayan_tensor_buffer(factors);
    std::vector<float> factor_values;
    if (status == 0 && factor_buffer == nullptr) {
        factor_values.assign((size_t) n_dims / 2, 1.0f);
        status = ggml_backend_dayan_buffer_create(ctx->context, factor_values.size() * sizeof(float), &buffers.input2);
        factor_buffer = buffers.input2;
    }
    if (status == 0 && factors == nullptr) {
        status = ggml_backend_dayan_upload(ctx->stream, buffers.input2, factor_values.data(),
                                           factor_values.size() * sizeof(float));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "rope", (uint64_t) config.batches * config.tokens * config.heads, &attrs);
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "residual-add", ggml_backend_dayan_native_tiles(config.elements), &attrs);
        if (add->type == GGML_TYPE_F32) {
            status = residual_add_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(add->src[0]),
                                         ggml_backend_dayan_tensor_buffer(add->src[1]),
                                         ggml_backend_dayan_tensor_buffer(add), &attrs);
        } else if (add->type == GGML_TYPE_BF16) {
            status = residual_add_bf16_1c(buffers.config, ggml_backend_dayan_tensor_buffer(add->src[0]),
                                          ggml_backend_dayan_tensor_buffer(add->src[1]),
                                          ggml_backend_dayan_tensor_buffer(add), &attrs);
        } else {
            status = residual_add_fp16_1c(buffers.config, ggml_backend_dayan_tensor_buffer(add->src[0]),
                                          ggml_backend_dayan_tensor_buffer(add->src[1]),
                                          ggml_backend_dayan_tensor_buffer(add), &attrs);
        }
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(ctx, rms ? "rms-norm" : "norm", config.rows, &attrs);
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
        0,
        0,
    };
    const float epsilon = ggml_get_op_params_f32(output, 1);
    std::memcpy(&config.epsilon_bits, &epsilon, sizeof(epsilon));
    const uint32_t channels_per_group = (config.channels + config.groups - 1U) / config.groups;
    const uint32_t last_channels = (config.channels - 1U) % channels_per_group + 1U;
    const float inv_group = 1.0f / ((float) config.width * config.height * channels_per_group);
    const float inv_last_group = 1.0f / ((float) config.width * config.height * last_channels);
    std::memcpy(&config.inv_group_bits, &inv_group, sizeof(inv_group));
    std::memcpy(&config.inv_last_group_bits, &inv_last_group, sizeof(inv_last_group));

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, "group-norm", (uint64_t) config.batches * config.groups, &attrs);
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
        has_add ? kDayanNormHasResidual : 0U,
        kDayanDataTypeF32,
    };
    if ((slot.output_mask & kDayanNormSlotOutputResidual) != 0U) {
        config.flags |= kDayanNormWriteResidual;
    }
    if ((slot.output_mask & kDayanNormSlotOutputRms) != 0U) {
        config.flags |= kDayanNormWriteRms;
    }
    if ((slot.output_mask & kDayanNormSlotOutputScaled) != 0U) {
        config.flags |= kDayanNormWriteScaled;
    }
    if ((slot.output_mask & kDayanNormSlotOutputLowp) != 0U) {
        if (lowp == nullptr) {
            return -1;
        }
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_native_launch_attrs(
            ctx, has_add ? "add-rms" : "rms-scale", config.norm.rows, &attrs);
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
    if (q->ne[1] == 4 && q->ne[0] == 256) {
        GGML_LOG_INFO("flash cfg batch=%u qh=%u kvh=%u qlen=%u kvlen=%u off=%u flags=%u qnb=[%zu,%zu,%zu,%zu] knb=[%zu,%zu,%zu,%zu] vnb=[%zu,%zu,%zu,%zu]\\n",
                      config.batch, config.q_heads, config.kv_heads, config.q_len, config.kv_len,
                      config.causal_offset, config.flags, q->nb[0], q->nb[1], q->nb[2], q->nb[3],
                      k->nb[0], k->nb[1], k->nb[2], k->nb[3], v->nb[0], v->nb[1], v->nb[2], v->nb[3]);
    }

    ggml_backend_dayan_buffers buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), sizeof(int64_t), &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        const bool split_kv = ctx->device->topology.clusters && dayan_flash_attention_use_split_kv(config.q_len, config.kv_len);
        const bool four_core = ctx->device->topology.clusters && (config.q_len > 64 || split_kv);
        const char * name = q->type == GGML_TYPE_BF16 ? "flash-attn-bf16" : "flash-attn-fp16";
        if (four_core) {
            ggml_backend_dayan_init_matmul_launch_attrs(ctx, name, &attrs);
        } else {
            ggml_backend_dayan_init_native_launch_attrs(ctx, name, (uint64_t) config.batch * config.q_heads, &attrs);
        }
        if (q->type == GGML_TYPE_BF16 || q->type == GGML_TYPE_F16) {
            const auto kernel = q->type == GGML_TYPE_BF16 ?
                (split_kv ? flash_attention_decode_bf16_4c : four_core ? flash_attention_bf16_4c : flash_attention_bf16_1c) :
                (split_kv ? flash_attention_decode_fp16_4c : four_core ? flash_attention_fp16_4c : flash_attention_fp16_1c);
            status = kernel(buffers.config, ggml_backend_dayan_tensor_buffer(q), ggml_backend_dayan_tensor_buffer(k),
                            ggml_backend_dayan_tensor_buffer(v), ggml_backend_dayan_tensor_buffer(dst), buffers.result, &attrs);
        } else {
            status = -1;
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
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ggml-gdn", &attrs);
        attrs.config = ctx->device->topology.launch(false, (uint64_t) config.heads * config.sequences);
        const auto kernel = config.gate_width == 1 && config.snapshots == 1 && config.tokens >= 16 ?
                                gated_delta_net_chunk_f32_1c : gated_delta_net_f32_1c;
        status = kernel(buffers.config, ggml_backend_dayan_tensor_buffer(q),
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
    npu_buffer_t *             qg_buffer = ggml_backend_dayan_tensor_buffer(qg);
    ggml_backend_dayan_buffers buffers;
    int status = qg_buffer != nullptr && (qg->type == GGML_TYPE_F32 || qg->type == weight->type) ? 0 : -1;
    if (status == 0) {
        status = ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    }
    if (status == 0) {
        status = ggml_backend_dayan_prepare_slot_workspace(ctx, (size_t) tokens * width * sizeof(uint16_t) * (ctx->device->topology.clusters ? ctx->device->topology.clusters : ctx->device->topology.cores), &buffers);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_matmul_launch_attrs(ctx, "gate-out", &attrs);
        if (qg->type == GGML_TYPE_F32) {
            status = weight->type == GGML_TYPE_BF16 ?
                         ggml_backend_dayan_matmul_kernel(ctx, project_gated_output_f32_gate_bf16_1c, project_gated_output_f32_gate_bf16_4c)(
                             buffers.config, ggml_backend_dayan_tensor_buffer(attention), qg_buffer,
                             ggml_backend_dayan_tensor_buffer(weight), ggml_backend_dayan_tensor_buffer(projection),
                             buffers.input2, &attrs) :
                         ggml_backend_dayan_matmul_kernel(ctx, project_gated_output_f32_gate_fp16_1c, project_gated_output_f32_gate_fp16_4c)(
                             buffers.config, ggml_backend_dayan_tensor_buffer(attention), qg_buffer,
                             ggml_backend_dayan_tensor_buffer(weight), ggml_backend_dayan_tensor_buffer(projection),
                             buffers.input2, &attrs);
        } else {
            status = weight->type == GGML_TYPE_BF16 ?
                         ggml_backend_dayan_matmul_kernel(ctx, project_gated_output_bf16_1c, project_gated_output_bf16_4c)(
                             buffers.config, ggml_backend_dayan_tensor_buffer(attention), qg_buffer,
                             ggml_backend_dayan_tensor_buffer(weight), ggml_backend_dayan_tensor_buffer(projection),
                             buffers.input2, &attrs) :
                         ggml_backend_dayan_matmul_kernel(ctx, project_gated_output_fp16_1c, project_gated_output_fp16_4c)(
                             buffers.config, ggml_backend_dayan_tensor_buffer(attention), qg_buffer,
                             ggml_backend_dayan_tensor_buffer(weight), ggml_backend_dayan_tensor_buffer(projection),
                             buffers.input2, &attrs);
        }
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
    DayanSSMConvConfig config = {
        (uint32_t) input->ne[0],  (uint32_t) input->ne[1],
        (uint32_t) output->ne[1], (uint32_t) input->ne[2],
        (uint32_t) weight->ne[0], slot.kind == ggml_backend_dayan_slot_kind::ssm_conv_silu ? 1U : 0U, 0, 0,
    };
    const float neg_log2e = -1.4426950408889634f;
    const float one       = 1.0f;
    std::memcpy(&config.neg_log2e_bits, &neg_log2e, sizeof(neg_log2e));
    std::memcpy(&config.one_bits, &one, sizeof(one));
    ggml_backend_dayan_buffers buffers;
    int                        status =
        ggml_backend_dayan_prepare_launch_buffers(ctx, &config, sizeof(config), 0, &buffers);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        ggml_backend_dayan_init_launch_attrs(ctx, "ssm-conv", &attrs);
        attrs.config = ctx->device->topology.launch(false, dayan_ssm_conv_task_count(config.channels, config.tokens, config.sequences, ctx->device->topology.cores));
        status = ssm_conv_f32_1c(buffers.config, ggml_backend_dayan_tensor_buffer(input),
                                 ggml_backend_dayan_tensor_buffer(weight), ggml_backend_dayan_tensor_buffer(output), &attrs);
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
    for (ggml_backend_dayan_slot_resources & resources : ctx->slot_resources) {
        if (resources.workspace != nullptr) {
            npu_buffer_destroy(resources.workspace);
        }
        if (resources.result != nullptr) {
            npu_buffer_destroy(resources.result);
        }
        if (resources.config != nullptr) {
            npu_buffer_destroy(resources.config);
        }
    }
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
    for (auto * extra : context->mxfp4) {
        delete extra;
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
    if (tensor->type == GGML_TYPE_MXFP4) {
        const auto layout = DayanMxfp4Layout::make(ggml_nrows(tensor), tensor->ne[0]);
        npu_buffer_t * scales = nullptr;
        if (npu_buffer_create_view(view, layout.scale_offset, layout.bytes - layout.scale_offset, &scales) < 0) {
            tensor->extra = nullptr;
            return GGML_STATUS_FAILED;
        }
        context->views.push_back(scales);
        auto * extra = new ggml_backend_dayan_mxfp4_extra{view, scales, layout};
        context->mxfp4.push_back(extra);
        tensor->extra = extra;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_dayan_mxfp4_transfer(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                            void * data, size_t offset, size_t size, bool write) {
    auto * context = (ggml_backend_dayan_buffer_context *) buffer->context;
    auto * extra = (ggml_backend_dayan_mxfp4_extra *) tensor->extra;
    GGML_ASSERT(extra != nullptr && offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    std::vector<uint8_t> device(extra->layout.bytes, 0);
    std::vector<uint8_t> logical(ggml_nbytes(tensor));
    if (!write || offset != 0 || size != logical.size()) {
        npu_event_t * event = nullptr;
        GGML_ASSERT(npu_memcpy_d2h_async(context->device->stream, device.data(), extra->data, 0, device.size(), &event) == 0);
        GGML_ASSERT(ggml_backend_dayan_event_wait(event) == 0);
        dayan_mxfp4_unpack(extra->layout, device.data(), logical.data());
    }
    if (write) {
        std::memcpy(logical.data() + offset, data, size);
        dayan_mxfp4_pack(extra->layout, logical.data(), device.data());
        npu_event_t * event = nullptr;
        GGML_ASSERT(npu_memcpy_h2d_async(context->device->stream, extra->data, 0, device.data(), device.size(), &event) == 0);
        GGML_ASSERT(ggml_backend_dayan_event_wait(event) == 0);
    } else {
        std::memcpy(data, logical.data() + offset, size);
    }
}

static void ggml_backend_dayan_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                 ggml_tensor *         tensor,
                                                 const void *          data,
                                                 size_t                offset,
                                                 size_t                size) {
    if (size == 0) {
        return;
    }
    if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_backend_dayan_mxfp4_transfer(buffer, tensor, const_cast<void *>(data), offset, size, true);
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
    if (std::getenv("DAYAN_TRACE_SET") != nullptr) {
        std::fprintf(stderr, "DAYAN_SET t=%llu name=%s offset=%zu size=%zu dev=%p\n",
                     (unsigned long long) ggml_time_us(), tensor->name, offset, size, (void *) view);
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
    if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_backend_dayan_mxfp4_transfer(buffer, tensor, data, offset, size, false);
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
    if (tensor->type == GGML_TYPE_MXFP4) {
        std::vector<uint8_t> data(size, value);
        ggml_backend_dayan_mxfp4_transfer(buffer, tensor, data.data(), offset, size, true);
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
    if (src->type == GGML_TYPE_MXFP4 || dst->type == GGML_TYPE_MXFP4) {
        if (src->type != dst->type || !ggml_are_same_shape(src, dst)) {
            return false;
        }
        const auto * extra = (const ggml_backend_dayan_mxfp4_extra *) src->extra;
        return ggml_backend_dayan_copy(context->device->stream, ggml_backend_dayan_tensor_buffer(dst), 0,
                                      extra->data, 0, extra->layout.bytes) == 0;
    }
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

static size_t ggml_backend_dayan_buffer_type_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_MXFP4 ?
        DayanMxfp4Layout::make(ggml_nrows(tensor), tensor->ne[0]).bytes : ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_dayan_buffer_type_i = {
    /* .get_name       = */ ggml_backend_dayan_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_dayan_buffer_type_alloc,
    /* .get_alignment  = */ ggml_backend_dayan_buffer_type_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ ggml_backend_dayan_buffer_type_alloc_size,
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

static bool ggml_backend_dayan_mega_resolve(void *,
                                             const void * pointer,
                                             size_t size,
                                             npu_buffer_t ** buffer,
                                             uint64_t * offset) {
    ggml_backend_dayan_device_pointer result;
    if (!ggml_backend_dayan_find_device_pointer(pointer, size, &result)) {
        return false;
    }
    *buffer = result.buffer;
    *offset = result.offset;
    return true;
}

static int ggml_backend_dayan_mega_launch(void * context,
                                          uint16_t graph,
                                          const void * config,
                                          size_t config_size,
                                          npu_buffer_t * const * roots) {
    ggml_backend_dayan_context * ctx = (ggml_backend_dayan_context *) context;
    ggml_backend_dayan_buffers  buffers;
    int status = ggml_backend_dayan_prepare_launch_buffers(ctx, config, config_size, 0, &buffers);
    npu_buffer_t * arguments[kDayanMegaRootCount];
    for (size_t index = 0; index < kDayanMegaRootCount; ++index) {
        arguments[index] = roots[index] != nullptr ? roots[index] : buffers.config;
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        const bool linear = graph == kDayanMegaQwen35DenseMxfp4Linear ||
                            graph == kDayanMegaQwen35DenseBf16Linear;
        ggml_backend_dayan_init_launch_attrs(
            ctx, linear ? "q35-linear" : "q35-full-mega", &attrs);
        attrs.config = { NPUTaskDim3{ 4, 1, 1 }, NPUTaskMode::kFourCore };
        if (std::getenv("DAYAN_TRACE_SET") != nullptr) {
            std::fprintf(stderr, "MEGA_BEGIN t=%llu\n", (unsigned long long) ggml_time_us());
        }
        status = qwen35_dense_mega(
            buffers.config, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5],
            arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12],
            arguments[13], arguments[14], arguments[15], &attrs);
        if (std::getenv("DAYAN_TRACE_SET") != nullptr) {
            std::fprintf(stderr, "MEGA_END t=%llu status=%d\n", (unsigned long long) ggml_time_us(), status);
        }
    }
    ggml_backend_dayan_buffers_free(&buffers, &status);
    return status;
}

static bool ggml_backend_dayan_mega_enabled(const ggml_backend_dayan_context * ctx) {
    const char * enabled = std::getenv("GGML_DAYAN_MEGA");
    if (enabled != nullptr && strcmp(enabled, "0") == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(ggml_backend_dayan_trace_mutex);
    return ctx->device->topology.clusters == 1 && ctx->device->topology.cores == 4 &&
           ggml_backend_dayan_trace_callback == nullptr && ggml_backend_dayan_filter_callback == nullptr;
}

static enum ggml_status ggml_backend_dayan_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_dayan_context *  ctx = (ggml_backend_dayan_context *) backend->context;
    ggml_backend_dayan_graph_plan plan;
    const ggml_tensor *           unsupported      = nullptr;
    const int64_t                 compile_start_us = ggml_time_us();

    if (!ggml_backend_dayan_compile_graph(cgraph, &plan, &unsupported)) {
        GGML_LOG_ERROR("%s: unsupported op %s after %.3f ms\n", __func__, ggml_op_desc(unsupported),
                       (double) (ggml_time_us() - compile_start_us) / 1000.0);
        return GGML_STATUS_FAILED;
    }
    const double compile_ms = (double) (ggml_time_us() - compile_start_us) / 1000.0;

    ggml_backend_dayan_graph_buffers graph_buffers;
    int                              status = 0;

    if (ctx->slot_resources.size() < plan.slots.size()) {
        ctx->slot_resources.resize(plan.slots.size());
    }

    const ggml_backend_dayan_mega_runtime mega_runtime = {
        ctx,
        ggml_backend_dayan_mega_enabled(ctx),
        ggml_backend_dayan_mega_resolve,
        ggml_backend_dayan_mega_launch,
    };
    bool split_boundary = false;

    for (size_t slot_index = 0; slot_index < plan.slots.size(); ++slot_index) {
        ctx->active_slot = slot_index;
        const ggml_backend_dayan_mega_result mega =
            ggml_backend_dayan_try_qwen35_mega(mega_runtime, plan.slots, slot_index);
        if (mega.launched) {
            ++ggml_backend_dayan_mega_launch_count;
            status = mega.status;
            if (status < 0) {
                GGML_LOG_ERROR("%s: slots %zu..%zu failed in Qwen3.5 mega executor with status %d\n", __func__,
                               slot_index, mega.end, status);
                break;
            }
            split_boundary = true;
            slot_index     = mega.end - 1;
            continue;
        }

        ggml_backend_dayan_graph_slot boundary_slot;
        const ggml_backend_dayan_graph_slot * slot_ptr = &plan.slots[slot_index];
        if (split_boundary) {
            boundary_slot              = *slot_ptr;
            boundary_slot.nodes[0]     = nullptr;
            boundary_slot.output_mask &= ~kDayanNormSlotOutputResidual;
            slot_ptr                    = &boundary_slot;
            split_boundary              = false;
        }
        const ggml_backend_dayan_graph_slot & slot = *slot_ptr;
        if (!ggml_backend_dayan_filter_slot(slot, slot_index)) {
            ggml_backend_dayan_trace_slot(slot, slot_index, false, 0, 0);
            continue;
        }
        const int64_t                         slot_start_us = ggml_time_us();
        if (ggml_backend_dayan_is_mxfp4_slot(slot)) {
            status = ggml_backend_dayan_mxfp4_slot(ctx, &graph_buffers, slot);
        } else switch (slot.kind) {
            case ggml_backend_dayan_slot_kind::mul_mat:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat(
                            ctx, slot.nodes[0], slot.nodes[0], activation_buffer, false);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::mul_mat_silu:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat(
                            ctx, slot.nodes[0], slot.nodes[1], activation_buffer, true);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::mul_mat_pair:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat_pair(ctx, slot, activation_buffer);
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
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat_id(ctx, slot.nodes[0], activation_buffer);
                    }
                }
                break;
            case ggml_backend_dayan_slot_kind::mul_mat_id_pair:
                {
                    npu_buffer_t * activation_buffer = nullptr;
                    status = ggml_backend_dayan_prepare_activation(ctx, &graph_buffers, slot.nodes[0]->src[1],
                                                                   slot.nodes[0]->src[0]->type, &activation_buffer);
                    if (status == 0) {
                        status = ggml_backend_dayan_mul_mat_id_pair(ctx, slot, activation_buffer);
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
                status = ggml_backend_dayan_gated_output_projection(ctx, slot);
                break;
            case ggml_backend_dayan_slot_kind::ssm_conv:
            case ggml_backend_dayan_slot_kind::ssm_conv_silu:
                status = ggml_backend_dayan_ssm_conv(ctx, slot);
                break;
        }
        ggml_backend_dayan_trace_slot(slot, slot_index, true, status, ggml_time_us() - slot_start_us);
        if (status < 0) {
            GGML_LOG_ERROR("%s: slot %zu (%s, %s) failed with status %d\n", __func__, slot_index,
                           ggml_op_desc(slot.nodes[0]), slot.nodes[0]->name, status);
            break;
        }
    }
    ctx->active_slot = SIZE_MAX;

    ggml_backend_dayan_graph_buffers_free(&graph_buffers, &status);

    if (status < 0) {
        GGML_LOG_ERROR("%s: graph execution failed with status %d\n", __func__, status);
        return GGML_STATUS_FAILED;
    }

    if (!plan.slots.empty()) {
        size_t   memory_bf16_widen_slots    = 0;
        size_t   memory_f16_widen_slots     = 0;
        size_t   local_bf16_widen_slots     = 0;
        size_t   local_f16_widen_slots      = 0;
        uint64_t memory_bf16_widen_elements = 0;
        uint64_t memory_f16_widen_elements  = 0;
        uint64_t local_bf16_widen_elements  = 0;
        uint64_t local_f16_widen_elements   = 0;
        for (const ggml_backend_dayan_graph_slot & slot : plan.slots) {
            enum ggml_type source_type = GGML_TYPE_COUNT;
            uint64_t       elements    = 0;
            bool           local       = false;
            if ((slot.kind == ggml_backend_dayan_slot_kind::get_rows ||
                 slot.kind == ggml_backend_dayan_slot_kind::copy) &&
                slot.nodes[0] != nullptr && slot.nodes[0]->src[0] != nullptr &&
                slot.nodes[0]->type == GGML_TYPE_F32) {
                source_type = slot.nodes[0]->src[0]->type;
                elements    = ggml_nelements(slot.nodes[0]);
            } else if (slot.kind == ggml_backend_dayan_slot_kind::softmax &&
                       slot.nodes[0] != nullptr && slot.nodes[0]->src[1] != nullptr) {
                source_type = slot.nodes[0]->src[1]->type;
                elements    = ggml_nelements(slot.nodes[0]);
                local       = true;
            } else if (slot.kind == ggml_backend_dayan_slot_kind::gated_output_projection &&
                       slot.nodes[0] != nullptr && slot.nodes[3] != nullptr) {
                source_type = slot.nodes[3]->type;
                elements    = ggml_nelements(slot.nodes[0]);
                local       = true;
            } else if (slot.kind == ggml_backend_dayan_slot_kind::norm_pipeline && slot.nodes[1] != nullptr &&
                       slot.nodes[1]->src[0] != nullptr) {
                source_type = slot.nodes[1]->src[0]->type;
                elements    = ggml_nelements(slot.nodes[1]);
                local       = true;
            } else if (slot.kind == ggml_backend_dayan_slot_kind::rms_norm && slot.nodes[0] != nullptr &&
                       slot.nodes[0]->src[0] != nullptr) {
                source_type = slot.nodes[0]->src[0]->type;
                elements    = ggml_nelements(slot.nodes[0]);
                local       = true;
            }
            if (source_type == GGML_TYPE_BF16) {
                if (local) {
                    ++local_bf16_widen_slots;
                    local_bf16_widen_elements += elements;
                } else {
                    ++memory_bf16_widen_slots;
                    memory_bf16_widen_elements += elements;
                }
            } else if (source_type == GGML_TYPE_F16) {
                if (local) {
                    ++local_f16_widen_slots;
                    local_f16_widen_elements += elements;
                } else {
                    ++memory_f16_widen_slots;
                    memory_f16_widen_elements += elements;
                }
            }
        }
        GGML_LOG_DEBUG(
            "%s: lowered %d nodes to %zu slots, paired=%zu, fused=%zu, residual=%zu, rms=%zu, "
            "attention=%zu, delta=%zu, gated=%zu, conv=%zu, moe=%zu, moe-fused=%zu, routing=%zu, activations=%zu, "
            "memory-widen=bf16:%zu/%" PRIu64 ",f16:%zu/%" PRIu64 ", "
            "local-widen=bf16:%zu/%" PRIu64 ",f16:%zu/%" PRIu64 ", "
            "runtime-cast=bf16:%zu/%" PRIu64 ",f16:%zu/%" PRIu64 ", compile=%.3f ms\n",
            __func__, cgraph->n_nodes, plan.slots.size(), plan.paired_mul_mat_slots, plan.fused_mul_mat_slots,
            plan.residual_add_slots, plan.rms_norm_slots, plan.flash_attention_slots, plan.gated_delta_net_slots,
            plan.gated_output_slots, plan.ssm_conv_slots, plan.moe_slots, plan.fused_moe_slots, plan.routing_slots,
            graph_buffers.activation_by_tensor.size(), memory_bf16_widen_slots, memory_bf16_widen_elements,
            memory_f16_widen_slots, memory_f16_widen_elements, local_bf16_widen_slots, local_bf16_widen_elements,
            local_f16_widen_slots, local_f16_widen_elements, graph_buffers.runtime_bf16_casts,
            graph_buffers.runtime_bf16_cast_elements, graph_buffers.runtime_f16_casts,
            graph_buffers.runtime_f16_cast_elements, compile_ms);
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

    if (ggml_is_empty(op)) {
        return true;
    }

    switch (op->op) {
        case GGML_OP_NONE:
            return true;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return op->type != GGML_TYPE_MXFP4 ||
                   (op->view_src != nullptr && op->view_offs == 0 && ggml_is_contiguous(op) &&
                    op->ne[0] == op->view_src->ne[0] && ggml_nrows(op) == ggml_nrows(op->view_src));
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
    if (strcmp(name, "ggml_backend_dayan_set_slot_trace_callback") == 0) {
        return (void *) ggml_backend_dayan_set_slot_trace_callback;
    }
    if (strcmp(name, "ggml_backend_dayan_set_slot_filter_callback") == 0) {
        return (void *) ggml_backend_dayan_set_slot_filter_callback;
    }
    if (strcmp(name, "ggml_backend_dayan_get_graph_plan") == 0) {
        return (void *) ggml_backend_dayan_get_graph_plan;
    }
    if (strcmp(name, "ggml_backend_dayan_get_mega_launch_count") == 0) {
        return (void *) ggml_backend_dayan_get_mega_launch_count;
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
