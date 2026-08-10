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

bool ggml_backend_dayan_supports_residual_add(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_are_same_shape(op, op->src[1])) {
        return false;
    }

    return ggml_nelements(op) > 0 && ggml_nelements(op) <= 0x7ffff000LL && ggml_is_contiguous(op) &&
           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
}

static bool ggml_backend_dayan_supports_elementwise_binary(const ggml_tensor * op, enum ggml_op kind) {
    if (op == nullptr || op->op != kind || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_can_repeat(op->src[1], op->src[0])) {
        return false;
    }

    return ggml_nelements(op) > 0 && (uint64_t) ggml_nelements(op) <= 0x7ffff000ULL;
}

bool ggml_backend_dayan_supports_elementwise_add(const ggml_tensor * op) {
    return ggml_backend_dayan_supports_elementwise_binary(op, GGML_OP_ADD);
}

bool ggml_backend_dayan_supports_elementwise_mul(const ggml_tensor * op) {
    return ggml_backend_dayan_supports_elementwise_binary(op, GGML_OP_MUL);
}

bool ggml_backend_dayan_supports_elementwise_sub(const ggml_tensor * op) {
    return ggml_backend_dayan_supports_elementwise_binary(op, GGML_OP_SUB);
}

bool ggml_backend_dayan_supports_elementwise_div(const ggml_tensor * op) {
    return ggml_backend_dayan_supports_elementwise_binary(op, GGML_OP_DIV);
}

bool ggml_backend_dayan_supports_scale(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SCALE || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) ||
        ggml_nelements(op) <= 0 || (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }

    float parameters[2];
    std::memcpy(parameters, op->op_params, sizeof(parameters));
    return std::isfinite(parameters[0]) && std::isfinite(parameters[1]);
}

bool ggml_backend_dayan_supports_unary(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) ||
        ggml_nelements(op) <= 0 || (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }

    const enum ggml_unary_op unary = ggml_get_unary_op(op);
    return unary == GGML_UNARY_OP_SIGMOID || unary == GGML_UNARY_OP_SILU || unary == GGML_UNARY_OP_SOFTPLUS;
}

bool ggml_backend_dayan_supports_math_unary(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || ggml_nelements(op) <= 0 ||
        (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return true;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_EXP:
                    return true;
                default:
                    return false;
            }
        default:
            return false;
    }
}

bool ggml_backend_dayan_supports_reduce(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[0]->ne[0] <= 0 || op->src[0]->ne[0] > UINT32_MAX || ggml_nrows(op->src[0]) <= 0 ||
        ggml_nrows(op->src[0]) > UINT32_MAX) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_SUM:
            return ggml_is_scalar(op) && ggml_nelements(op->src[0]) <= UINT32_MAX;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return op->ne[0] == 1 && op->ne[1] == op->src[0]->ne[1] && op->ne[2] == op->src[0]->ne[2] &&
                   op->ne[3] == op->src[0]->ne[3];
        default:
            return false;
    }
}

bool ggml_backend_dayan_supports_l2_norm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_L2_NORM || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || op->ne[0] <= 0 ||
        op->ne[0] > UINT32_MAX || ggml_nrows(op) <= 0 || ggml_nrows(op) > UINT32_MAX) {
        return false;
    }

    float epsilon;
    std::memcpy(&epsilon, op->op_params, sizeof(epsilon));
    return std::isfinite(epsilon) && epsilon >= 0.0f;
}

static bool ggml_backend_dayan_is_copy_type(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16;
}

bool ggml_backend_dayan_supports_get_rows(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 ||
        (op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_BF16) ||
        (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64) ||
        op->ne[0] != op->src[0]->ne[0] || op->ne[1] != op->src[1]->ne[0] || op->ne[2] != op->src[1]->ne[1] ||
        op->ne[3] != op->src[1]->ne[2]) {
        return false;
    }

    return op->ne[0] > 0 && op->ne[0] <= UINT32_MAX && ggml_nrows(op) >= 0 && ggml_nrows(op) <= UINT32_MAX;
}

bool ggml_backend_dayan_supports_set_rows(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] == nullptr || op->src[0]->type != GGML_TYPE_F32 ||
        (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64) ||
        !ggml_backend_dayan_is_copy_type(op->type) || op->type != op->src[2]->type ||
        op->ne[0] != op->src[0]->ne[0] || !ggml_are_same_shape(op, op->src[2])) {
        return false;
    }

    return op->ne[0] > 0 && op->ne[0] <= UINT32_MAX && ggml_nrows(op->src[0]) > 0 &&
           ggml_nrows(op->src[0]) <= UINT32_MAX;
}

bool ggml_backend_dayan_supports_copy(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_CPY && op->src[0] != nullptr &&
           ggml_backend_dayan_is_copy_type(op->src[0]->type) && ggml_backend_dayan_is_copy_type(op->type) &&
           ggml_nelements(op) == ggml_nelements(op->src[0]) && ggml_nelements(op) >= 0 &&
           (uint64_t) ggml_nelements(op) <= 0x7ffff000ULL;
}

bool ggml_backend_dayan_supports_layout_copy(const ggml_tensor * op) {
    return op != nullptr && (op->op == GGML_OP_DUP || op->op == GGML_OP_CONT) && op->src[0] != nullptr &&
           ggml_backend_dayan_is_copy_type(op->type) && op->type == op->src[0]->type &&
           ggml_nelements(op) == ggml_nelements(op->src[0]) && ggml_nelements(op) >= 0 &&
           (uint64_t) ggml_nelements(op) <= 0x7ffff000ULL;
}

bool ggml_backend_dayan_supports_repeat(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_REPEAT || op->src[0] == nullptr || op->type != op->src[0]->type ||
        !ggml_backend_dayan_is_copy_type(op->type) || !ggml_can_repeat(op->src[0], op) || ggml_nelements(op) < 0 ||
        (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }

    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (op->src[0]->ne[dimension] < 0 || op->src[0]->ne[dimension] > UINT32_MAX || op->ne[dimension] < 0 ||
            op->ne[dimension] > UINT32_MAX) {
            return false;
        }
    }
    return true;
}

bool ggml_backend_dayan_supports_set(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SET || op->src[0] == nullptr || op->src[1] == nullptr ||
        (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_I32) || op->src[0]->type != op->type ||
        op->src[1]->type != op->type || !ggml_are_same_shape(op, op->src[0]) || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous(op->src[0]) || ggml_nelements(op->src[1]) <= 0) {
        return false;
    }

    const int32_t * parameters = op->op_params;
    if (parameters[0] < 0 || parameters[1] < 0 || parameters[2] < 0 || parameters[3] < 0) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (op->src[1]->ne[dimension] <= 0 || op->src[1]->ne[dimension] > UINT32_MAX) {
            return false;
        }
    }

    const uint64_t element_bytes = ggml_type_size(op->type);
    const uint64_t last = (uint64_t) parameters[3] +
                          (uint64_t) (op->src[1]->ne[0] - 1) * element_bytes +
                          (uint64_t) (op->src[1]->ne[1] - 1) * (uint32_t) parameters[0] +
                          (uint64_t) (op->src[1]->ne[2] - 1) * (uint32_t) parameters[1] +
                          (uint64_t) (op->src[1]->ne[3] - 1) * (uint32_t) parameters[2] + element_bytes;
    return last <= ggml_nbytes(op);
}

bool ggml_backend_dayan_supports_layout(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || ggml_nelements(op) <= 0 ||
        (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (op->src[0]->ne[dimension] <= 0 || op->src[0]->ne[dimension] > UINT32_MAX || op->ne[dimension] <= 0 ||
            op->ne[dimension] > UINT32_MAX) {
            return false;
        }
    }

    switch (op->op) {
        case GGML_OP_DIAG:
            return op->src[0]->ne[1] == 1 && op->ne[0] == op->src[0]->ne[0] &&
                   op->ne[1] == op->src[0]->ne[0] && op->ne[2] == op->src[0]->ne[2] &&
                   op->ne[3] == op->src[0]->ne[3];
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG_MASK_ZERO:
            return ggml_are_same_shape(op, op->src[0]) && ggml_get_op_params_i32(op, 0) >= 0;
        case GGML_OP_PAD:
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                const int64_t left  = ggml_get_op_params_i32(op, 2 * dimension);
                const int64_t right = ggml_get_op_params_i32(op, 2 * dimension + 1);
                if (op->ne[dimension] != op->src[0]->ne[dimension] + left + right) {
                    return false;
                }
            }
            return ggml_get_op_params_i32(op, 8) == 0 || ggml_get_op_params_i32(op, 8) == 1;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_get_op_params_i32(op, 0) >= 0 && ggml_get_op_params_i32(op, 1) >= 0 &&
                   ggml_get_op_params_i32(op, 0) < op->src[0]->ne[0] &&
                   ggml_get_op_params_i32(op, 1) < op->src[0]->ne[0] &&
                   op->ne[0] == op->src[0]->ne[0] + ggml_get_op_params_i32(op, 0) +
                                    ggml_get_op_params_i32(op, 1) &&
                   op->ne[1] == op->src[0]->ne[1] && op->ne[2] == op->src[0]->ne[2] &&
                   op->ne[3] == op->src[0]->ne[3];
        case GGML_OP_ROLL:
            if (!ggml_are_same_shape(op, op->src[0])) {
                return false;
            }
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                const int64_t shift = ggml_get_op_params_i32(op, dimension);
                if (shift <= -op->ne[dimension] || shift >= op->ne[dimension]) {
                    return false;
                }
            }
            return true;
        default:
            return false;
    }
}

bool ggml_backend_dayan_supports_clamp(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CLAMP || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || ggml_nelements(op) <= 0 ||
        (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }

    float parameters[2];
    std::memcpy(parameters, op->op_params, sizeof(parameters));
    return std::isfinite(parameters[0]) && std::isfinite(parameters[1]) && parameters[0] <= parameters[1];
}

bool ggml_backend_dayan_supports_softmax(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SOFT_MAX || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous(op->src[0])) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (op->ne[dimension] <= 0 || op->ne[dimension] > UINT32_MAX) {
            return false;
        }
    }

    float parameters[2];
    std::memcpy(parameters, op->op_params, sizeof(parameters));
    if (!std::isfinite(parameters[0]) || !std::isfinite(parameters[1])) {
        return false;
    }

    const ggml_tensor * mask = op->src[1];
    if (mask != nullptr &&
        ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) || !ggml_is_contiguous(mask) ||
         mask->ne[0] != op->ne[0] || mask->ne[1] < op->ne[1] || mask->ne[2] <= 0 ||
         op->ne[2] % mask->ne[2] != 0 || mask->ne[3] <= 0 || op->ne[3] % mask->ne[3] != 0 ||
         mask->ne[1] > UINT32_MAX || mask->ne[2] > UINT32_MAX || mask->ne[3] > UINT32_MAX)) {
        return false;
    }
    if (parameters[1] > 0.0f && mask == nullptr) {
        return false;
    }

    const ggml_tensor * sinks = op->src[2];
    return sinks == nullptr ||
           (sinks->type == GGML_TYPE_F32 && ggml_is_contiguous(sinks) && ggml_is_vector(sinks) &&
            ggml_nelements(sinks) == op->ne[2]);
}

bool ggml_backend_dayan_supports_concat(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CONCAT || op->src[0] == nullptr || op->src[1] == nullptr ||
        !ggml_backend_dayan_is_copy_type(op->type) || op->src[0]->type != op->type || op->src[1]->type != op->type) {
        return false;
    }

    const int32_t axis = ggml_get_op_params_i32(op, 0);
    if (axis < 0 || axis >= GGML_MAX_DIMS || ggml_nelements(op) <= 0 ||
        (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (op->src[0]->ne[dimension] <= 0 || op->src[0]->ne[dimension] > UINT32_MAX ||
            op->src[1]->ne[dimension] <= 0 || op->src[1]->ne[dimension] > UINT32_MAX || op->ne[dimension] <= 0 ||
            op->ne[dimension] > UINT32_MAX ||
            (dimension == axis ? op->ne[dimension] != op->src[0]->ne[dimension] + op->src[1]->ne[dimension] :
                                 (op->ne[dimension] != op->src[0]->ne[dimension] ||
                                  op->ne[dimension] != op->src[1]->ne[dimension]))) {
            return false;
        }
    }
    return true;
}

bool ggml_backend_dayan_supports_rope(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ROPE || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_I32 ||
        !ggml_are_same_shape(op, op->src[0]) || op->ne[0] <= 0 || op->ne[0] > UINT32_MAX || op->ne[1] <= 0 ||
        op->ne[1] > UINT32_MAX || op->ne[2] <= 0 || op->ne[2] > UINT32_MAX || op->ne[3] <= 0 ||
        op->ne[3] > UINT32_MAX) {
        return false;
    }

    const int32_t n_dims = ggml_get_op_params_i32(op, 1);
    const int32_t mode   = ggml_get_op_params_i32(op, 2);
    if (n_dims <= 0 || n_dims > op->ne[0] || (n_dims & 1) != 0 ||
        (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_MROPE &&
         mode != GGML_ROPE_TYPE_IMROPE)) {
        return false;
    }

    const int64_t position_axes = (mode & GGML_ROPE_TYPE_MROPE) != 0 ? 4 : 1;
    if (!ggml_is_vector(op->src[1]) || ggml_nelements(op->src[1]) != position_axes * op->ne[2]) {
        return false;
    }
    if (op->src[2] != nullptr &&
        (op->src[2]->type != GGML_TYPE_F32 || ggml_nelements(op->src[2]) < n_dims / 2)) {
        return false;
    }

    float parameters[6];
    std::memcpy(parameters, (const int32_t *) op->op_params + 5, sizeof(parameters));
    return std::isfinite(parameters[0]) && parameters[0] > 0.0f && std::isfinite(parameters[1]) &&
           parameters[1] > 0.0f && std::isfinite(parameters[2]) && std::isfinite(parameters[3]) &&
           std::isfinite(parameters[4]) && std::isfinite(parameters[5]);
}

bool ggml_backend_dayan_supports_rms_norm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || op->ne[0] <= 0 ||
        op->ne[0] > UINT32_MAX ||
        ggml_nrows(op) <= 0 || ggml_nrows(op) > UINT32_MAX) {
        return false;
    }

    float epsilon;
    std::memcpy(&epsilon, op->op_params, sizeof(epsilon));
    return std::isfinite(epsilon) && epsilon >= 0.0f;
}

bool ggml_backend_dayan_supports_norm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_NORM || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || op->ne[0] <= 0 ||
        op->ne[0] > UINT32_MAX || ggml_nrows(op) <= 0 || ggml_nrows(op) > UINT32_MAX) {
        return false;
    }

    float epsilon;
    std::memcpy(&epsilon, op->op_params, sizeof(epsilon));
    return std::isfinite(epsilon) && epsilon >= 0.0f;
}

bool ggml_backend_dayan_supports_group_norm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GROUP_NORM || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || op->ne[0] <= 0 ||
        op->ne[0] > UINT32_MAX || op->ne[1] <= 0 || op->ne[1] > UINT32_MAX || op->ne[2] <= 0 ||
        op->ne[2] > UINT32_MAX || op->ne[3] <= 0 || op->ne[3] > UINT32_MAX) {
        return false;
    }

    const int32_t groups = ggml_get_op_params_i32(op, 0);
    const float   epsilon = ggml_get_op_params_f32(op, 1);
    return groups > 0 && groups <= op->ne[2] && std::isfinite(epsilon) && epsilon >= 0.0f;
}

bool ggml_backend_dayan_supports_rms_norm_scale(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }

    const bool left_norm  = ggml_backend_dayan_supports_rms_norm(op->src[0]);
    const bool right_norm = ggml_backend_dayan_supports_rms_norm(op->src[1]);
    if (left_norm == right_norm) {
        return false;
    }

    const ggml_tensor * norm  = left_norm ? op->src[0] : op->src[1];
    const ggml_tensor * scale = left_norm ? op->src[1] : op->src[0];
    return ggml_is_contiguous(norm) && ggml_is_contiguous(norm->src[0]) && scale->type == GGML_TYPE_F32 &&
           scale->ne[0] == norm->ne[0] && scale->ne[1] == 1 && scale->ne[2] == 1 && scale->ne[3] == 1 &&
           ggml_is_contiguous(scale) && ggml_are_same_shape(op, norm);
}

bool ggml_backend_dayan_supports_quantized_rms_norm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CPY || op->src[0] == nullptr ||
        !ggml_backend_dayan_supports_rms_norm_scale(op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }

    return ggml_backend_dayan_is_storage_type(op->type) && ggml_are_same_shape(op, op->src[0]);
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
    std::memcpy(&max_bias, (const char *) op->op_params + sizeof(scale), sizeof(max_bias));
    std::memcpy(&logit_softcap,
                (const char *) op->op_params + sizeof(scale) + sizeof(max_bias),
                sizeof(logit_softcap));

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

bool ggml_backend_dayan_supports_gated_delta_net(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GATED_DELTA_NET || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] == nullptr || op->src[3] == nullptr || op->src[4] == nullptr || op->src[5] == nullptr ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor * q         = op->src[0];
    const ggml_tensor * k         = op->src[1];
    const ggml_tensor * v         = op->src[2];
    const ggml_tensor * g         = op->src[3];
    const ggml_tensor * beta      = op->src[4];
    const ggml_tensor * state     = op->src[5];
    const int32_t       snapshots = ggml_get_op_params_i32(op, 0);
    const int64_t       head_dim  = v->ne[0];
    const int64_t       heads     = v->ne[1];
    const int64_t       tokens    = v->ne[2];
    const int64_t       sequences = v->ne[3];

    return q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32 &&
           g->type == GGML_TYPE_F32 && beta->type == GGML_TYPE_F32 && state->type == GGML_TYPE_F32 && head_dim == 128 &&
           heads > 0 && heads <= UINT32_MAX && tokens > 0 && tokens <= UINT32_MAX && sequences > 0 &&
           sequences <= UINT32_MAX && snapshots > 0 && ggml_is_contiguous_rows(q) && ggml_is_contiguous_rows(k) &&
           ggml_is_contiguous_rows(v) && ggml_is_contiguous(g) && ggml_is_contiguous(beta) &&
           ggml_is_contiguous(state) && q->ne[0] == head_dim && k->ne[0] == head_dim && q->ne[1] > 0 &&
           heads % q->ne[1] == 0 && k->ne[1] > 0 && heads % k->ne[1] == 0 && q->ne[2] == tokens && k->ne[2] == tokens &&
           q->ne[3] > 0 && sequences % q->ne[3] == 0 && k->ne[3] > 0 && sequences % k->ne[3] == 0 &&
           (g->ne[0] == 1 || g->ne[0] == head_dim) && g->ne[1] == heads && g->ne[2] == tokens &&
           g->ne[3] == sequences && beta->ne[0] == 1 && beta->ne[1] == heads && beta->ne[2] == tokens &&
           beta->ne[3] == sequences && state->ne[0] == head_dim && state->ne[1] == head_dim && state->ne[2] == heads &&
           state->ne[3] == sequences && op->ne[0] == head_dim * heads &&
           op->ne[1] == tokens * sequences + (int64_t) snapshots * head_dim * sequences && op->ne[2] == 1 &&
           op->ne[3] == 1;
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
            return true;
        case GGML_OP_CONT:
            return ggml_backend_dayan_supports_gate_cont(node);
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

static bool ggml_backend_dayan_starts_fused_glu(const ggml_cgraph * cgraph, int matmul_index) {
    if (matmul_index >= cgraph->n_nodes || !ggml_backend_dayan_supports_mul_mat(cgraph->nodes[matmul_index])) {
        return false;
    }

    ggml_tensor * matmul    = cgraph->nodes[matmul_index];
    const int     pair_index = ggml_backend_dayan_next_compute(cgraph, matmul_index + 1);
    if (pair_index >= cgraph->n_nodes || !ggml_backend_dayan_supports_mul_mat(cgraph->nodes[pair_index])) {
        return false;
    }

    ggml_tensor * pair       = cgraph->nodes[pair_index];
    const int     glu_index  = ggml_backend_dayan_next_compute(cgraph, pair_index + 1);
    const int     cast_index = ggml_backend_dayan_next_compute(cgraph, glu_index + 1);
    return matmul->src[1] == pair->src[1] && matmul->src[0]->type == pair->src[0]->type &&
           glu_index < cgraph->n_nodes && cast_index < cgraph->n_nodes &&
           ggml_backend_dayan_supports_glu(cgraph->nodes[glu_index]) &&
           ggml_backend_dayan_supports_quantized_glu(cgraph->nodes[cast_index]) &&
           ((cgraph->nodes[glu_index]->src[0] == matmul && cgraph->nodes[glu_index]->src[1] == pair) ||
            (cgraph->nodes[glu_index]->src[0] == pair && cgraph->nodes[glu_index]->src[1] == matmul)) &&
           cgraph->nodes[cast_index]->src[0] == cgraph->nodes[glu_index];
}

bool ggml_backend_dayan_compile_graph(const ggml_cgraph *             cgraph,
                                      ggml_backend_dayan_graph_plan * plan,
                                      const ggml_tensor **            unsupported) {
    plan->slots.clear();
    plan->paired_mul_mat_slots  = 0;
    plan->fused_mul_mat_slots   = 0;
    plan->residual_add_slots    = 0;
    plan->rms_norm_slots        = 0;
    plan->rms_norm_matmul_slots = 0;
    plan->flash_attention_slots = 0;
    plan->gated_delta_net_slots = 0;
    plan->gated_output_slots    = 0;
    plan->ssm_conv_slots        = 0;
    *unsupported                = nullptr;

    for (int i = 0; i < cgraph->n_nodes;) {
        i = ggml_backend_dayan_next_compute(cgraph, i);
        if (i >= cgraph->n_nodes) {
            break;
        }

        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_backend_dayan_supports_residual_add(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::residual_add;
            slot.nodes[0]   = node;
            slot.node_count = 1;

            const int rms_index   = ggml_backend_dayan_next_compute(cgraph, i + 1);
            const int scale_index = ggml_backend_dayan_next_compute(cgraph, rms_index + 1);
            if (rms_index < cgraph->n_nodes && scale_index < cgraph->n_nodes &&
                ggml_backend_dayan_supports_rms_norm(cgraph->nodes[rms_index]) &&
                cgraph->nodes[rms_index]->src[0] == node &&
                ggml_backend_dayan_supports_rms_norm_scale(cgraph->nodes[scale_index]) &&
                (cgraph->nodes[scale_index]->src[0] == cgraph->nodes[rms_index] ||
                 cgraph->nodes[scale_index]->src[1] == cgraph->nodes[rms_index])) {
                slot.kind       = ggml_backend_dayan_slot_kind::add_rms_norm_scaled;
                slot.nodes[1]   = cgraph->nodes[rms_index];
                slot.nodes[2]   = cgraph->nodes[scale_index];
                slot.node_count = 3;
                i               = scale_index + 1;

                const int cast_index = ggml_backend_dayan_next_compute(cgraph, i);
                if (cast_index < cgraph->n_nodes &&
                    ggml_backend_dayan_supports_quantized_rms_norm(cgraph->nodes[cast_index]) &&
                    cgraph->nodes[cast_index]->src[0] == cgraph->nodes[scale_index]) {
                    slot.kind       = ggml_backend_dayan_slot_kind::add_rms_norm_quantized;
                    slot.nodes[3]   = cgraph->nodes[cast_index];
                    slot.node_count = 4;
                    i               = cast_index + 1;

                    const int matmul_index = ggml_backend_dayan_next_compute(cgraph, i);
                    if (matmul_index < cgraph->n_nodes &&
                        ggml_backend_dayan_supports_mul_mat(cgraph->nodes[matmul_index]) &&
                        cgraph->nodes[matmul_index]->src[1] == cgraph->nodes[cast_index] &&
                        !ggml_backend_dayan_starts_fused_glu(cgraph, matmul_index)) {
                        slot.kind       = ggml_backend_dayan_slot_kind::add_rms_norm_mul_mat;
                        slot.nodes[4]   = cgraph->nodes[matmul_index];
                        slot.node_count = 5;
                        i               = matmul_index + 1;
                        ++plan->rms_norm_matmul_slots;
                    }
                }
                ++plan->rms_norm_slots;
                ++plan->fused_mul_mat_slots;
            } else {
                ++i;
            }
            plan->slots.push_back(slot);
            ++plan->residual_add_slots;
            continue;
        }
        if (ggml_backend_dayan_supports_rms_norm(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::rms_norm;
            slot.nodes[0]   = node;
            slot.node_count = 1;

            const int scale_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
            if (scale_index < cgraph->n_nodes &&
                ggml_backend_dayan_supports_rms_norm_scale(cgraph->nodes[scale_index]) &&
                (cgraph->nodes[scale_index]->src[0] == node || cgraph->nodes[scale_index]->src[1] == node)) {
                slot.kind       = ggml_backend_dayan_slot_kind::rms_norm_scaled;
                slot.nodes[1]   = cgraph->nodes[scale_index];
                slot.node_count = 2;
                i               = scale_index + 1;

                const int cast_index = ggml_backend_dayan_next_compute(cgraph, i);
                if (cast_index < cgraph->n_nodes &&
                    ggml_backend_dayan_supports_quantized_rms_norm(cgraph->nodes[cast_index]) &&
                    cgraph->nodes[cast_index]->src[0] == cgraph->nodes[scale_index]) {
                    slot.kind       = ggml_backend_dayan_slot_kind::rms_norm_quantized;
                    slot.nodes[2]   = cgraph->nodes[cast_index];
                    slot.node_count = 3;
                    i               = cast_index + 1;

                    const int matmul_index = ggml_backend_dayan_next_compute(cgraph, i);
                    if (matmul_index < cgraph->n_nodes &&
                        ggml_backend_dayan_supports_mul_mat(cgraph->nodes[matmul_index]) &&
                        cgraph->nodes[matmul_index]->src[1] == cgraph->nodes[cast_index] &&
                        !ggml_backend_dayan_starts_fused_glu(cgraph, matmul_index)) {
                        slot.kind       = ggml_backend_dayan_slot_kind::rms_norm_mul_mat;
                        slot.nodes[3]   = cgraph->nodes[matmul_index];
                        slot.node_count = 4;
                        i               = matmul_index + 1;
                        ++plan->rms_norm_matmul_slots;
                        ++plan->fused_mul_mat_slots;
                    }
                }
            } else {
                ++i;
            }
            plan->slots.push_back(slot);
            ++plan->rms_norm_slots;
            continue;
        }
        if (ggml_backend_dayan_supports_gated_delta_net(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::gated_delta_net;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++plan->gated_delta_net_slots;
            ++i;
            continue;
        }
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
        if (ggml_backend_dayan_supports_l2_norm(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::l2_norm;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_scale(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::scale;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_elementwise_add(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::elementwise_add;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_elementwise_mul(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::elementwise_mul;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_elementwise_sub(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::elementwise_sub;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_elementwise_div(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::elementwise_div;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_unary(node)) {
            ggml_backend_dayan_graph_slot slot;
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_SIGMOID:
                    slot.kind = ggml_backend_dayan_slot_kind::sigmoid;
                    break;
                case GGML_UNARY_OP_SILU:
                    slot.kind = ggml_backend_dayan_slot_kind::silu;
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    slot.kind = ggml_backend_dayan_slot_kind::softplus;
                    break;
                default:
                    *unsupported = node;
                    return false;
            }
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_math_unary(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::unary_math;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_reduce(node)) {
            ggml_backend_dayan_graph_slot slot;
            switch (node->op) {
                case GGML_OP_SUM:
                    slot.kind = ggml_backend_dayan_slot_kind::sum;
                    break;
                case GGML_OP_SUM_ROWS:
                    slot.kind = ggml_backend_dayan_slot_kind::sum_rows;
                    break;
                case GGML_OP_MEAN:
                    slot.kind = ggml_backend_dayan_slot_kind::mean;
                    break;
                default:
                    *unsupported = node;
                    return false;
            }
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_get_rows(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::get_rows;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_set_rows(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::set_rows;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_copy(node) || ggml_backend_dayan_supports_layout_copy(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::copy;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_repeat(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::repeat;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_set(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::set;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_layout(node)) {
            ggml_backend_dayan_graph_slot slot;
            switch (node->op) {
                case GGML_OP_DIAG:
                    slot.kind = ggml_backend_dayan_slot_kind::diag;
                    break;
                case GGML_OP_DIAG_MASK_INF:
                    slot.kind = ggml_backend_dayan_slot_kind::diag_mask_inf;
                    break;
                case GGML_OP_DIAG_MASK_ZERO:
                    slot.kind = ggml_backend_dayan_slot_kind::diag_mask_zero;
                    break;
                case GGML_OP_PAD:
                    slot.kind = ggml_get_op_params_i32(node, 8) == 0 ? ggml_backend_dayan_slot_kind::pad :
                                                                      ggml_backend_dayan_slot_kind::pad_circular;
                    break;
                case GGML_OP_PAD_REFLECT_1D:
                    slot.kind = ggml_backend_dayan_slot_kind::pad_reflect_1d;
                    break;
                case GGML_OP_ROLL:
                    slot.kind = ggml_backend_dayan_slot_kind::roll;
                    break;
                default:
                    *unsupported = node;
                    return false;
            }
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_clamp(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::clamp;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_softmax(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::softmax;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_concat(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::concat;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_rope(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::rope;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_norm(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::norm;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_group_norm(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::group_norm;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
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
