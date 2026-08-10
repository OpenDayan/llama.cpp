#include "ggml-dayan-graph.h"

#include "ggml-dayan.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

static bool ggml_backend_dayan_is_storage_type(enum ggml_type type) {
    return type == GGML_TYPE_BF16 || type == GGML_TYPE_F16;
}

static bool ggml_backend_dayan_is_flash_type(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16;
}

bool ggml_backend_dayan_supports_mul_mat(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * weight          = op->src[0];
    const ggml_tensor * activation      = op->src[1];
    const int64_t       dimension_limit = 0x7ffff000LL;

    return weight != nullptr && activation != nullptr && ggml_backend_dayan_is_storage_type(weight->type) &&
           (activation->type == GGML_TYPE_F32 || activation->type == weight->type) && op->type == GGML_TYPE_F32 &&
           weight->ne[0] == activation->ne[0] && op->ne[0] == weight->ne[1] && op->ne[1] == activation->ne[1] &&
           weight->ne[0] > 0 && weight->ne[0] <= dimension_limit && weight->ne[1] > 0 &&
           weight->ne[1] <= dimension_limit && activation->ne[1] > 0 && activation->ne[1] <= dimension_limit &&
           weight->ne[1] % 16 == 0 && weight->ne[2] == 1 && weight->ne[3] == 1 && activation->ne[2] == 1 &&
           activation->ne[3] == 1 && op->ne[2] == 1 && op->ne[3] == 1 && ggml_is_contiguous(weight) &&
           ggml_is_contiguous(activation) && ggml_is_contiguous(op);
}

bool ggml_backend_dayan_supports_glu(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GLU || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor *    gate = op->src[0];
    const ggml_tensor *    up   = op->src[1];
    const enum ggml_glu_op glu  = ggml_get_glu_op(op);

    return (glu == GGML_GLU_OP_SWIGLU || glu == GGML_GLU_OP_GEGLU) && ggml_backend_dayan_supports_mul_mat(gate) &&
           ggml_backend_dayan_supports_mul_mat(up) && gate->src[1] == up->src[1] &&
           gate->src[0]->type == up->src[0]->type && ggml_are_same_shape(gate, up) && ggml_are_same_shape(op, gate);
}

bool ggml_backend_dayan_supports_quantized_mul_mat(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CPY || op->src[0] == nullptr ||
        !ggml_backend_dayan_supports_mul_mat(op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor * mul_mat = op->src[0];
    return op->type == mul_mat->src[0]->type && ggml_are_same_shape(op, mul_mat);
}

bool ggml_backend_dayan_supports_quantized_glu(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CPY || op->src[0] == nullptr ||
        !ggml_backend_dayan_supports_glu(op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor * glu = op->src[0];
    return op->type == glu->src[0]->src[0]->type && ggml_are_same_shape(op, glu);
}

bool ggml_backend_dayan_flash_attention_lengths(const ggml_tensor * op,
                                                uint32_t *          causal_offset,
                                                uint32_t *          kv_len) {
    if (op == nullptr || causal_offset == nullptr || kv_len == nullptr) {
        return false;
    }

    const ggml_tensor * mask = op->src[3];
    if (mask == nullptr) {
        if (op->src[1] == nullptr || op->src[1]->ne[1] <= 0 || op->src[1]->ne[1] > UINT32_MAX) {
            return false;
        }
        *causal_offset = 0;
        *kv_len         = (uint32_t) op->src[1]->ne[1];
        return true;
    }
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    if (q == nullptr || k == nullptr || mask->data == nullptr || mask->type != GGML_TYPE_F16 ||
        mask->ne[0] != k->ne[1] || mask->ne[1] != q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        !ggml_is_contiguous_rows(mask) || k->ne[1] < q->ne[1]) {
        return false;
    }

    int64_t offset = -1;
    for (int64_t query = 0; query < q->ne[1]; ++query) {
        int64_t valid_count = 0;
        bool    masked_tail = false;
        for (int64_t key = 0; key < k->ne[1]; ++key) {
            const char * address = (const char *) mask->data + query * mask->nb[1] + key * mask->nb[0];
            ggml_fp16_t value_f16;
            std::memcpy(&value_f16, address, sizeof(value_f16));
            const float value  = ggml_fp16_to_fp32(value_f16);
            const bool  masked = (std::isinf(value) && value < 0.0f) || (std::isfinite(value) && value <= -1.0e4f);
            if (!masked_tail && value == 0.0f) {
                ++valid_count;
                continue;
            }
            if (!masked) {
                return false;
            }
            masked_tail = true;
        }

        const int64_t query_offset = valid_count - query - 1;
        if (query_offset < 0 || query_offset > UINT32_MAX || (offset >= 0 && offset != query_offset)) {
            return false;
        }
        offset = query_offset;
    }

    const uint64_t effective_kv_len = (uint64_t) offset + (uint64_t) q->ne[1];
    if (effective_kv_len == 0 || effective_kv_len > (uint64_t) k->ne[1] || effective_kv_len > UINT32_MAX) {
        return false;
    }

    *causal_offset = (uint32_t) offset;
    *kv_len         = (uint32_t) effective_kv_len;
    return true;
}

bool ggml_backend_dayan_supports_flash_attention(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_FLASH_ATTN_EXT || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] == nullptr || op->src[4] != nullptr || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    float               scale;
    float               max_bias;
    float               logit_softcap;
    std::memcpy(&scale, op->op_params, sizeof(scale));
    std::memcpy(&max_bias, op->op_params + sizeof(scale), sizeof(max_bias));
    std::memcpy(&logit_softcap, op->op_params + sizeof(scale) + sizeof(max_bias), sizeof(logit_softcap));

    const int64_t dimension_limit = std::numeric_limits<uint32_t>::max();
    const ggml_tensor * mask = op->src[3];
    const bool mask_supported = mask == nullptr ||
                                (mask->type == GGML_TYPE_F16 && mask->ne[0] == k->ne[1] &&
                                 mask->ne[1] == q->ne[1] && mask->ne[2] == 1 && mask->ne[3] == 1 &&
                                 ggml_is_contiguous_rows(mask));
    return ggml_backend_dayan_is_flash_type(q->type) && ggml_backend_dayan_is_flash_type(k->type) &&
           ggml_backend_dayan_is_flash_type(v->type) && ggml_is_contiguous_rows(q) &&
           ggml_is_contiguous_rows(k) && ggml_is_contiguous_rows(v) && q->ne[0] == k->ne[0] &&
           q->ne[0] == v->ne[0] && q->ne[0] > 0 && q->ne[0] <= 256 && q->ne[0] % 64 == 0 && q->ne[1] > 0 &&
           q->ne[1] <= dimension_limit && k->ne[1] > 0 && k->ne[1] <= dimension_limit && q->ne[2] > 0 &&
           q->ne[2] <= dimension_limit && k->ne[2] > 0 && k->ne[2] <= dimension_limit && q->ne[2] % k->ne[2] == 0 &&
           q->ne[3] > 0 && q->ne[3] <= dimension_limit && k->ne[3] == q->ne[3] && v->ne[1] == k->ne[1] &&
           v->ne[2] == k->ne[2] && v->ne[3] == k->ne[3] && op->ne[0] == v->ne[0] && op->ne[1] == q->ne[2] &&
           op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3] && std::isfinite(scale) && scale >= 0.0f &&
           max_bias == 0.0f && logit_softcap == 0.0f && mask_supported;
}

bool ggml_backend_dayan_supports_gate_cont(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CONT || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(op) || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }

    const ggml_tensor * view = op->src[0];
    const ggml_tensor * qg   = view->view_src != nullptr ? view->view_src : view->src[0];
    return view->op == GGML_OP_VIEW && qg != nullptr && ggml_backend_dayan_supports_mul_mat(qg) &&
           view->type == GGML_TYPE_F32 && view->ne[0] > 0 && view->ne[1] > 0 && view->ne[2] == qg->ne[1] &&
           view->ne[3] == 1 && op->ne[0] == view->ne[0] * view->ne[1] && op->ne[1] == view->ne[2] &&
           qg->ne[0] == 2 * op->ne[0] && view->nb[1] == 2 * view->ne[0] * sizeof(float) &&
           view->nb[2] == 2 * op->ne[0] * sizeof(float) && view->view_offs == view->ne[0] * sizeof(float);
}

bool ggml_backend_dayan_supports_gate_sigmoid(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID &&
           op->src[0] != nullptr && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
           ggml_backend_dayan_supports_gate_cont(op->src[0]) && ggml_are_same_shape(op, op->src[0]);
}

bool ggml_backend_dayan_supports_gated_attention(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    const bool left_gate  = ggml_backend_dayan_supports_gate_sigmoid(op->src[0]);
    const bool right_gate = ggml_backend_dayan_supports_gate_sigmoid(op->src[1]);
    if (left_gate == right_gate) {
        return false;
    }
    const ggml_tensor * attention = left_gate ? op->src[1] : op->src[0];
    return attention->type == GGML_TYPE_F32 && ggml_is_contiguous(attention) && ggml_are_same_shape(op, attention);
}

bool ggml_backend_dayan_supports_ssm_conv(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SSM_CONV || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
        return false;
    }
    const ggml_tensor * input  = op->src[0];
    const ggml_tensor * weight = op->src[1];
    return input->ne[0] > 0 && input->ne[1] > 0 && input->ne[2] > 0 && input->ne[3] == 1 && weight->ne[0] > 0 &&
           weight->ne[1] == input->ne[1] && weight->ne[2] == 1 && weight->ne[3] == 1 &&
           input->ne[0] == weight->ne[0] - 1 + op->ne[1] && op->ne[0] == input->ne[1] && op->ne[2] == input->ne[2] &&
           op->ne[3] == 1 && input->ne[0] <= UINT32_MAX && input->ne[1] <= UINT32_MAX && input->ne[2] <= UINT32_MAX &&
           weight->ne[0] <= UINT32_MAX && op->ne[1] <= UINT32_MAX;
}

bool ggml_backend_dayan_supports_ssm_conv_silu(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SILU &&
           op->src[0] != nullptr && ggml_backend_dayan_supports_ssm_conv(op->src[0]) && op->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op, op->src[0]);
}

struct ggml_backend_dayan_cast_key {
    ggml_tensor *  activation;
    enum ggml_type type;

    bool operator==(const ggml_backend_dayan_cast_key & other) const {
        return activation == other.activation && type == other.type;
    }
};

struct ggml_backend_dayan_cast_key_hash {
    size_t operator()(const ggml_backend_dayan_cast_key & key) const {
        return std::hash<ggml_tensor *>{}(key.activation) ^ ((size_t) key.type << 1);
    }
};

bool ggml_backend_dayan_graph_rewrite(ggml_backend_t backend, ggml_context * ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    if (ctx == nullptr || cgraph == nullptr) {
        return false;
    }

    std::vector<ggml_tensor *> nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    std::vector<std::pair<ggml_tensor *, ggml_backend_dayan_cast_key>>                               matmuls;
    std::vector<ggml_backend_dayan_cast_key>                                                         cast_keys;
    std::unordered_map<ggml_backend_dayan_cast_key, ggml_tensor *, ggml_backend_dayan_cast_key_hash> casts;

    for (ggml_tensor * node : nodes) {
        if (ggml_backend_dayan_supports_mul_mat(node) && node->src[1]->type == GGML_TYPE_F32) {
            if (ggml_backend_dayan_supports_gated_attention(node->src[1])) {
                continue;
            }
            const ggml_backend_dayan_cast_key key = { node->src[1], node->src[0]->type };
            matmuls.emplace_back(node, key);
            if (casts.emplace(key, nullptr).second) {
                cast_keys.push_back(key);
            }
        }
    }

    if (cast_keys.empty()) {
        return true;
    }

    if (cgraph->n_nodes + (int) cast_keys.size() > ggml_graph_size(cgraph)) {
        GGML_LOG_WARN("%s: graph has no room for %zu low-precision activation casts\n", __func__, cast_keys.size());
        return false;
    }

    for (const ggml_backend_dayan_cast_key & key : cast_keys) {
        ggml_tensor * cast = ggml_cast(ctx, key.activation, key.type);
        cast->src[1]       = nullptr;  // keep scheduler assignment based on the source
        if (key.activation->name[0] != '\0') {
            ggml_format_name(cast, "%s (dayan %s)", key.activation->name, ggml_type_name(key.type));
        } else {
            ggml_format_name(cast, "dayan %s activation", ggml_type_name(key.type));
        }
        casts[key] = cast;
    }

    for (const auto & matmul : matmuls) {
        matmul.first->src[1] = casts.at(matmul.second);
    }

    ggml_graph_clear(cgraph);
    for (ggml_tensor * node : nodes) {
        ggml_build_forward_expand(cgraph, node);
    }

    GGML_LOG_DEBUG("%s: rewrote %zu matrix multiplications with %zu low-precision casts\n", __func__, matmuls.size(),
                   cast_keys.size());
    return true;
}

static bool ggml_backend_dayan_is_metadata(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
            return true;
        default:
            return false;
    }
}

static int ggml_backend_dayan_next_compute(const ggml_cgraph * cgraph, int index) {
    while (index < cgraph->n_nodes && ((cgraph->nodes[index]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                                       ggml_backend_dayan_is_metadata(cgraph->nodes[index]))) {
        ++index;
    }
    return index;
}

bool ggml_backend_dayan_compile_graph(const ggml_cgraph *             cgraph,
                                      ggml_backend_dayan_graph_plan * plan,
                                      const ggml_tensor **            unsupported) {
    plan->slots.clear();
    plan->paired_mul_mat_slots  = 0;
    plan->fused_mul_mat_slots   = 0;
    plan->flash_attention_slots = 0;
    plan->gated_output_slots    = 0;
    plan->ssm_conv_slots        = 0;
    *unsupported                = nullptr;

    for (int i = 0; i < cgraph->n_nodes;) {
        i = ggml_backend_dayan_next_compute(cgraph, i);
        if (i >= cgraph->n_nodes) {
            break;
        }

        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_backend_dayan_supports_ssm_conv(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind            = ggml_backend_dayan_slot_kind::ssm_conv;
            slot.nodes[0]        = node;
            slot.node_count      = 1;
            const int silu_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
            if (silu_index < cgraph->n_nodes && ggml_backend_dayan_supports_ssm_conv_silu(cgraph->nodes[silu_index]) &&
                cgraph->nodes[silu_index]->src[0] == node) {
                slot.kind       = ggml_backend_dayan_slot_kind::ssm_conv_silu;
                slot.nodes[1]   = cgraph->nodes[silu_index];
                slot.node_count = 2;
                i               = silu_index + 1;
            } else {
                ++i;
            }
            plan->slots.push_back(slot);
            ++plan->ssm_conv_slots;
            continue;
        }
        if (ggml_backend_dayan_supports_gate_sigmoid(node)) {
            const int mul_index        = ggml_backend_dayan_next_compute(cgraph, i + 1);
            const int projection_index = ggml_backend_dayan_next_compute(cgraph, mul_index + 1);
            if (mul_index >= cgraph->n_nodes || projection_index >= cgraph->n_nodes ||
                !ggml_backend_dayan_supports_gated_attention(cgraph->nodes[mul_index]) ||
                !ggml_backend_dayan_supports_mul_mat(cgraph->nodes[projection_index]) ||
                cgraph->nodes[projection_index]->src[1] != cgraph->nodes[mul_index]) {
                *unsupported = node;
                return false;
            }
            ggml_tensor *                 gate_cont = node->src[0];
            ggml_tensor *                 gate_view = gate_cont->src[0];
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::gated_output_projection;
            slot.nodes[0]   = node;
            slot.nodes[1]   = cgraph->nodes[mul_index];
            slot.nodes[2]   = cgraph->nodes[projection_index];
            slot.nodes[3]   = gate_view->view_src != nullptr ? gate_view->view_src : gate_view->src[0];
            slot.node_count = 3;
            plan->slots.push_back(slot);
            ++plan->fused_mul_mat_slots;
            ++plan->gated_output_slots;
            i = projection_index + 1;
            continue;
        }
        if (ggml_backend_dayan_supports_flash_attention(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::flash_attention;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++plan->flash_attention_slots;
            ++i;
            continue;
        }
        if (!ggml_backend_dayan_supports_mul_mat(node)) {
            *unsupported = node;
            return false;
        }

        ggml_backend_dayan_graph_slot slot;
        slot.nodes[0]   = node;
        slot.node_count = 1;

        const int quantized_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
        if (quantized_index < cgraph->n_nodes &&
            ggml_backend_dayan_supports_quantized_mul_mat(cgraph->nodes[quantized_index]) &&
            cgraph->nodes[quantized_index]->src[0] == node) {
            slot.kind       = ggml_backend_dayan_slot_kind::mul_mat_quantized;
            slot.nodes[1]   = cgraph->nodes[quantized_index];
            slot.node_count = 2;
            ++plan->fused_mul_mat_slots;
            i = quantized_index + 1;
            plan->slots.push_back(slot);
            continue;
        }

        const int pair_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
        if (pair_index < cgraph->n_nodes && ggml_backend_dayan_supports_mul_mat(cgraph->nodes[pair_index]) &&
            node->src[1] == cgraph->nodes[pair_index]->src[1] &&
            node->src[0]->type == cgraph->nodes[pair_index]->src[0]->type) {
            ggml_tensor * pair       = cgraph->nodes[pair_index];
            const int     glu_index  = ggml_backend_dayan_next_compute(cgraph, pair_index + 1);
            const int     cast_index = ggml_backend_dayan_next_compute(cgraph, glu_index + 1);

            if (glu_index < cgraph->n_nodes && cast_index < cgraph->n_nodes &&
                ggml_backend_dayan_supports_glu(cgraph->nodes[glu_index]) &&
                ggml_backend_dayan_supports_quantized_glu(cgraph->nodes[cast_index]) &&
                ((cgraph->nodes[glu_index]->src[0] == node && cgraph->nodes[glu_index]->src[1] == pair) ||
                 (cgraph->nodes[glu_index]->src[0] == pair && cgraph->nodes[glu_index]->src[1] == node)) &&
                cgraph->nodes[cast_index]->src[0] == cgraph->nodes[glu_index]) {
                ggml_tensor * glu = cgraph->nodes[glu_index];
                slot.kind = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU ? ggml_backend_dayan_slot_kind::mul_mat_swiglu :
                                                                         ggml_backend_dayan_slot_kind::mul_mat_geglu;
                slot.nodes[0]   = glu->src[0];
                slot.nodes[1]   = glu->src[1];
                slot.nodes[2]   = glu;
                slot.nodes[3]   = cgraph->nodes[cast_index];
                slot.node_count = 4;
                ++plan->fused_mul_mat_slots;
                i = cast_index + 1;
            } else {
                slot.kind       = ggml_backend_dayan_slot_kind::mul_mat_pair;
                slot.nodes[1]   = pair;
                slot.node_count = 2;
                ++plan->paired_mul_mat_slots;
                i = pair_index + 1;
            }
        } else {
            slot.kind = ggml_backend_dayan_slot_kind::mul_mat;
            ++i;
        }

        plan->slots.push_back(slot);
    }

    return true;
}
