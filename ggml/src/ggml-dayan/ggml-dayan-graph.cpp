/*
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include "ggml-dayan-graph.h"

#include "ggml-dayan.h"
#include "ggml-dayan-kernels.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

static bool ggml_backend_dayan_is_storage_type(enum ggml_type type) {
    return type == GGML_TYPE_BF16 || type == GGML_TYPE_F16;
}

static bool ggml_backend_dayan_3d_size_fits(size_t element_size, uint64_t ne0, uint64_t ne1, uint64_t ne2) {
    size_t         size          = element_size;
    const uint64_t dimensions[3] = { ne0, ne1, ne2 };
    for (int index = 0; index < 3; ++index) {
        if (dimensions[index] == 0 || dimensions[index] > SIZE_MAX / size) {
            return false;
        }
        size *= (size_t) dimensions[index];
    }
    return true;
}

static bool ggml_backend_dayan_4d_size_fits(
    size_t element_size, uint64_t ne0, uint64_t ne1, uint64_t ne2, uint64_t ne3) {
    if (!ggml_backend_dayan_3d_size_fits(element_size, ne0, ne1, ne2)) {
        return false;
    }
    const size_t size = element_size * (size_t) ne0 * (size_t) ne1 * (size_t) ne2;
    return ne3 > 0 && ne3 <= SIZE_MAX / size;
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

    const bool mx = weight != nullptr && weight->type == GGML_TYPE_MXFP4;
    const bool f32 = weight != nullptr && weight->type == GGML_TYPE_F32;
    if (f32 && (weight->ne[2] != 1 || weight->ne[3] != 1 || activation == nullptr || activation->type != GGML_TYPE_F32)) {
        return false;
    }
    if (mx && (weight->ne[0] % 32 != 0 || weight->ne[2] != 1 || weight->ne[3] != 1)) {
        return false;
    }
    if (weight == nullptr || activation == nullptr || (!mx && !f32 && !ggml_backend_dayan_is_storage_type(weight->type)) ||
        (activation->type != GGML_TYPE_F32 && activation->type != weight->type &&
         !(mx && ggml_backend_dayan_is_storage_type(activation->type))) || op->type != GGML_TYPE_F32 ||
        weight->ne[0] != activation->ne[0] || op->ne[0] != weight->ne[1] || op->ne[1] != activation->ne[1] ||
        op->ne[2] != activation->ne[2] || op->ne[3] != activation->ne[3] || weight->ne[0] <= 0 ||
        weight->ne[0] > dimension_limit || weight->ne[1] <= 0 || weight->ne[1] > dimension_limit ||
        activation->ne[1] <= 0 || activation->ne[1] > dimension_limit ||
        weight->ne[2] <= 0 || weight->ne[3] <= 0 || activation->ne[2] <= 0 || activation->ne[3] <= 0 ||
        activation->ne[2] % weight->ne[2] != 0 || activation->ne[3] % weight->ne[3] != 0 ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(activation) || !ggml_is_contiguous(op) ||
        !ggml_backend_dayan_4d_size_fits(
            sizeof(uint16_t), weight->ne[0], weight->ne[1], weight->ne[2], weight->ne[3]) ||
        !ggml_backend_dayan_4d_size_fits(
            sizeof(uint16_t), activation->ne[0], activation->ne[1], activation->ne[2], activation->ne[3]) ||
        !ggml_backend_dayan_4d_size_fits(
            sizeof(float), op->ne[0], op->ne[1], op->ne[2], op->ne[3])) {
        return false;
    }

    if (weight->ne[2] == 1 && weight->ne[3] == 1) {
        const uint64_t batches = (uint64_t) activation->ne[2] * (uint64_t) activation->ne[3];
        return batches <= (uint64_t) dimension_limit / (uint64_t) activation->ne[1];
    }
    return true;
}

static bool ggml_backend_dayan_mul_mat_has_single_batch(const ggml_tensor * op) {
    return op != nullptr && op->src[0] != nullptr && op->src[1] != nullptr && op->src[0]->ne[2] == 1 &&
           op->src[0]->ne[3] == 1 && op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1;
}

bool ggml_backend_dayan_supports_mul_mat_silu(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SILU &&
           op->src[0] != nullptr && ggml_backend_dayan_supports_mul_mat(op->src[0]) && op->type == GGML_TYPE_F32 &&
           op->src[0]->src[0]->type != GGML_TYPE_F32 &&
           ggml_is_contiguous(op) && ggml_are_same_shape(op, op->src[0]);
}

bool ggml_backend_dayan_supports_mul_mat_id(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT_ID || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] == nullptr) {
        return false;
    }

    const ggml_tensor * weight     = op->src[0];
    const ggml_tensor * activation = op->src[1];
    const ggml_tensor * ids        = op->src[2];
    const bool mx = weight->type == GGML_TYPE_MXFP4;
    if ((mx && weight->ne[0] % 32 != 0) || (!mx && !ggml_backend_dayan_is_storage_type(weight->type)) ||
        (activation->type != GGML_TYPE_F32 && activation->type != weight->type) || ids->type != GGML_TYPE_I32 ||
        op->type != GGML_TYPE_F32 || weight->ne[3] != 1 || activation->ne[3] != 1 || ids->ne[2] != 1 ||
        ids->ne[3] != 1 || op->ne[3] != 1 || weight->ne[0] != activation->ne[0] || ids->ne[1] != activation->ne[2] ||
        activation->ne[1] <= 0 || ids->ne[0] % activation->ne[1] != 0 || op->ne[0] != weight->ne[1] ||
        op->ne[1] != ids->ne[0] || op->ne[2] != ids->ne[1] || ids->nb[0] != sizeof(int32_t) ||
        ids->nb[1] < (uint64_t) ids->ne[0] * sizeof(int32_t) ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(activation) || !ggml_is_contiguous(op)) {
        return false;
    }

    const int64_t dimension_limit = 0x7ffff000LL;
    return weight->ne[0] > 0 && weight->ne[0] <= dimension_limit && weight->ne[1] > 0 &&
           weight->ne[1] <= dimension_limit && weight->ne[1] % 16 == 0 && weight->ne[2] > 0 &&
           weight->ne[2] <= INT32_MAX && activation->ne[1] <= UINT32_MAX && ids->ne[0] > 0 &&
           ids->ne[0] <= UINT32_MAX && ids->ne[1] > 0 && ids->ne[1] <= UINT32_MAX &&
           ggml_backend_dayan_3d_size_fits(sizeof(uint16_t), weight->ne[0], weight->ne[1], weight->ne[2]) &&
           ggml_backend_dayan_3d_size_fits(sizeof(uint16_t), activation->ne[0], activation->ne[1], activation->ne[2]) &&
           ggml_backend_dayan_3d_size_fits(sizeof(int32_t), ids->ne[0], ids->ne[1], 1) &&
           ggml_backend_dayan_3d_size_fits(sizeof(float), op->ne[0], op->ne[1], op->ne[2]);
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
           ggml_backend_dayan_supports_mul_mat(up) && gate->src[0]->type != GGML_TYPE_F32 && gate->src[1] == up->src[1] &&
           ggml_backend_dayan_mul_mat_has_single_batch(gate) &&
           ggml_backend_dayan_mul_mat_has_single_batch(up) && gate->src[0]->type == up->src[0]->type &&
           ggml_are_same_shape(gate, up) && ggml_are_same_shape(op, gate);
}

bool ggml_backend_dayan_supports_mul_mat_id_glu(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GLU || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor *    gate = op->src[0];
    const ggml_tensor *    up   = op->src[1];
    const enum ggml_glu_op glu  = ggml_get_glu_op(op);

    return (glu == GGML_GLU_OP_SWIGLU || glu == GGML_GLU_OP_GEGLU) && ggml_backend_dayan_supports_mul_mat_id(gate) &&
           ggml_backend_dayan_supports_mul_mat_id(up) && gate->src[1] == up->src[1] && gate->src[2] == up->src[2] &&
           gate->src[0]->type == up->src[0]->type && gate->src[0]->ne[2] == up->src[0]->ne[2] &&
           ggml_are_same_shape(gate, up) && ggml_are_same_shape(op, gate);
}

bool ggml_backend_dayan_supports_quantized_mul_mat(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CPY || op->src[0] == nullptr ||
        !ggml_backend_dayan_supports_mul_mat(op->src[0]) || op->src[0]->src[0]->type == GGML_TYPE_F32 ||
        !ggml_backend_dayan_mul_mat_has_single_batch(op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor * mul_mat = op->src[0];
    const bool mx = op->type == GGML_TYPE_MXFP4 && op->ne[0] % 32 == 0;
    const bool mx_input = mul_mat->src[0]->type == GGML_TYPE_MXFP4;
    return (mx || (mx_input && ggml_backend_dayan_is_storage_type(op->type)) ||
            (!mx_input && op->type == mul_mat->src[0]->type)) && ggml_are_same_shape(op, mul_mat);
}

bool ggml_backend_dayan_supports_quantized_glu(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CPY || op->src[0] == nullptr ||
        (!ggml_backend_dayan_supports_glu(op->src[0]) && !ggml_backend_dayan_supports_mul_mat_id_glu(op->src[0])) ||
        !ggml_is_contiguous(op)) {
        return false;
    }

    const ggml_tensor * glu = op->src[0];
    const bool mx = op->type == GGML_TYPE_MXFP4 && op->ne[0] % 32 == 0 &&
                    (glu->src[0]->op == GGML_OP_MUL_MAT || glu->src[0]->src[0]->type == GGML_TYPE_MXFP4);
    return (mx || (op->type != GGML_TYPE_MXFP4 && op->type == glu->src[0]->src[0]->type)) && ggml_are_same_shape(op, glu);
}

bool ggml_backend_dayan_supports_add_id(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD_ID || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] == nullptr) {
        return false;
    }

    const ggml_tensor * input = op->src[0];
    const ggml_tensor * bias  = op->src[1];
    const ggml_tensor * ids   = op->src[2];
    return op->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 &&
           ids->type == GGML_TYPE_I32 && input->ne[3] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1 && ids->ne[2] == 1 &&
           ids->ne[3] == 1 && input->ne[0] == bias->ne[0] && input->ne[1] == ids->ne[0] && input->ne[2] == ids->ne[1] &&
           ggml_are_same_shape(op, input) && ids->nb[0] == sizeof(int32_t) && input->ne[0] > 0 &&
           input->ne[0] <= UINT32_MAX && bias->ne[1] > 0 && bias->ne[1] <= INT32_MAX && input->ne[1] > 0 &&
           input->ne[1] <= UINT32_MAX && input->ne[2] > 0 && input->ne[2] <= UINT32_MAX &&
           ggml_backend_dayan_3d_size_fits(sizeof(float), input->ne[0], input->ne[1], input->ne[2]) &&
           ggml_backend_dayan_3d_size_fits(sizeof(float), bias->ne[0], bias->ne[1], 1) &&
           ggml_backend_dayan_3d_size_fits(sizeof(int32_t), ids->ne[0], ids->ne[1], 1) && ggml_is_contiguous(input) &&
           ggml_is_contiguous(bias) && ggml_is_contiguous(op);
}

static bool ggml_backend_dayan_supports_sort(const ggml_tensor * op, enum ggml_op kind) {
    if (op == nullptr || op->op != kind || op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_I32 || op->src[0]->ne[0] <= 0 || op->src[0]->ne[0] > kDayanSortMaxWidth || op->ne[0] <= 0 ||
        op->ne[0] > op->src[0]->ne[0] || ggml_nrows(op->src[0]) <= 0 || ggml_nrows(op->src[0]) > UINT32_MAX ||
        !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }

    return op->ne[1] == op->src[0]->ne[1] && op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3];
}

bool ggml_backend_dayan_supports_argsort(const ggml_tensor * op) {
    if (!ggml_backend_dayan_supports_sort(op, GGML_OP_ARGSORT) || !ggml_are_same_shape(op, op->src[0])) {
        return false;
    }
    const int32_t order = ggml_get_op_params_i32(op, 0);
    return order == GGML_SORT_ORDER_ASC || order == GGML_SORT_ORDER_DESC;
}

bool ggml_backend_dayan_supports_top_k(const ggml_tensor * op) {
    return ggml_backend_dayan_supports_sort(op, GGML_OP_TOP_K);
}

bool ggml_backend_dayan_supports_residual_add(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ADD || op->src[0] == nullptr || op->src[1] == nullptr ||
        (op->type != GGML_TYPE_F32 && !ggml_backend_dayan_is_storage_type(op->type)) ||
        op->src[0]->type != op->type || op->src[1]->type != op->type ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_are_same_shape(op, op->src[1])) {
        return false;
    }

    return ggml_nelements(op) > 0 && ggml_nelements(op) <= 0x7ffff000LL && ggml_is_contiguous(op) &&
           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
}

static bool ggml_backend_dayan_supports_elementwise_binary(const ggml_tensor * op, enum ggml_op kind) {
    if (op == nullptr || op->op != kind || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_can_repeat(op->src[1], op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }

    // A partial right row must fill whole vector lanes; the kernel rejects it
    // at launch time, so keep such shapes off the backend entirely.
    if (op->src[1]->ne[0] != op->ne[0] && (64U % op->src[1]->ne[0]) != 0U) {
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
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || ggml_nelements(op) < 0 ||
        (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }

    float parameters[2];
    std::memcpy(parameters, op->op_params, sizeof(parameters));
    return std::isfinite(parameters[0]) && std::isfinite(parameters[1]);
}

bool ggml_backend_dayan_supports_unary(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_UNARY || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || ggml_nelements(op) <= 0 ||
        (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
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
    const bool mx = op != nullptr && op->src[0] != nullptr && op->src[0]->type == GGML_TYPE_MXFP4;
    if (op == nullptr || op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        (!mx && !ggml_backend_dayan_is_copy_type(op->src[0]->type)) ||
        (mx ? (op->ne[0] % 32 != 0 || !ggml_backend_dayan_is_copy_type(op->type)) :
              (op->type != GGML_TYPE_F32 && op->type != op->src[0]->type)) ||
        (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64) || op->ne[0] != op->src[0]->ne[0] ||
        op->ne[1] != op->src[1]->ne[0] || op->ne[2] != op->src[1]->ne[1] || op->ne[3] != op->src[1]->ne[2] ||
        !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op)) {
        return false;
    }

    return op->ne[0] > 0 && op->ne[0] <= UINT32_MAX && ggml_nrows(op) >= 0 && ggml_nrows(op) <= UINT32_MAX;
}

bool ggml_backend_dayan_supports_set_rows(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->src[2] == nullptr || op->src[0]->type != GGML_TYPE_F32 ||
        (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64) ||
        !ggml_backend_dayan_is_copy_type(op->type) || op->type != op->src[2]->type || op->ne[0] != op->src[0]->ne[0] ||
        !ggml_are_same_shape(op, op->src[2])) {
        return false;
    }

    return op->ne[0] > 0 && op->ne[0] <= UINT32_MAX && ggml_nrows(op->src[0]) > 0 &&
           ggml_nrows(op->src[0]) <= UINT32_MAX;
}

bool ggml_backend_dayan_supports_copy(const ggml_tensor * op) {
    if (op != nullptr && op->op == GGML_OP_CPY && op->src[0] != nullptr && op->type == GGML_TYPE_MXFP4) {
        return ggml_backend_dayan_is_flash_type(op->src[0]->type) && op->ne[0] > 0 && op->ne[0] % 32 == 0 &&
               ggml_nelements(op) > 0 && (uint64_t) ggml_nelements(op) <= 0x7ffff000ULL &&
               ggml_are_same_shape(op, op->src[0]) && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
    }
    return op != nullptr && op->op == GGML_OP_CPY && op->src[0] != nullptr &&
           ggml_backend_dayan_is_copy_type(op->src[0]->type) && ggml_backend_dayan_is_copy_type(op->type) &&
           ggml_nelements(op) == ggml_nelements(op->src[0]) && ggml_nelements(op) >= 0 &&
           (uint64_t) ggml_nelements(op) <= 0x7ffff000ULL && ggml_is_contiguous(op);
}

bool ggml_backend_dayan_supports_layout_copy(const ggml_tensor * op) {
    return op != nullptr && (op->op == GGML_OP_DUP || op->op == GGML_OP_CONT) && op->src[0] != nullptr &&
           ggml_backend_dayan_is_copy_type(op->type) && op->type == op->src[0]->type &&
           ggml_nelements(op) == ggml_nelements(op->src[0]) && ggml_nelements(op) >= 0 &&
           (uint64_t) ggml_nelements(op) <= 0x7ffff000ULL && ggml_is_contiguous(op);
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
    const uint64_t last          = (uint64_t) parameters[3] + (uint64_t) (op->src[1]->ne[0] - 1) * element_bytes +
                                   (uint64_t) (op->src[1]->ne[1] - 1) * (uint32_t) parameters[0] +
                                   (uint64_t) (op->src[1]->ne[2] - 1) * (uint32_t) parameters[1] +
                                   (uint64_t) (op->src[1]->ne[3] - 1) * (uint32_t) parameters[2] + element_bytes;
    return last <= ggml_nbytes(op);
}

bool ggml_backend_dayan_supports_layout(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        ggml_nelements(op) <= 0 || (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
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
            return op->src[0]->ne[1] == 1 && op->ne[0] == op->src[0]->ne[0] && op->ne[1] == op->src[0]->ne[0] &&
                   op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3];
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
                   op->ne[0] == op->src[0]->ne[0] + ggml_get_op_params_i32(op, 0) + ggml_get_op_params_i32(op, 1) &&
                   op->ne[1] == op->src[0]->ne[1] && op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3];
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
    return parameters[0] <= parameters[1];
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
    if (mask != nullptr && ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) || !ggml_is_contiguous(mask) ||
                            mask->ne[0] != op->ne[0] || mask->ne[1] < op->ne[1] || mask->ne[2] <= 0 ||
                            op->ne[2] % mask->ne[2] != 0 || mask->ne[3] <= 0 || op->ne[3] % mask->ne[3] != 0 ||
                            mask->ne[1] > UINT32_MAX || mask->ne[2] > UINT32_MAX || mask->ne[3] > UINT32_MAX)) {
        return false;
    }
    if (parameters[1] > 0.0f && mask == nullptr) {
        return false;
    }

    const ggml_tensor * sinks = op->src[2];
    return sinks == nullptr || (sinks->type == GGML_TYPE_F32 && ggml_is_contiguous(sinks) && ggml_is_vector(sinks) &&
                                ggml_nelements(sinks) == op->ne[2]);
}

bool ggml_backend_dayan_supports_concat(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CONCAT || op->src[0] == nullptr || op->src[1] == nullptr ||
        !ggml_backend_dayan_is_copy_type(op->type) || op->src[0]->type != op->type || op->src[1]->type != op->type) {
        return false;
    }

    const int32_t axis = ggml_get_op_params_i32(op, 0);
    if (axis < 0 || axis >= GGML_MAX_DIMS || ggml_nelements(op) <= 0 || (uint64_t) ggml_nelements(op) > 0x7ffff000ULL) {
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (op->src[0]->ne[dimension] <= 0 || op->src[0]->ne[dimension] > UINT32_MAX ||
            op->src[1]->ne[dimension] <= 0 || op->src[1]->ne[dimension] > UINT32_MAX || op->ne[dimension] <= 0 ||
            op->ne[dimension] > UINT32_MAX ||
            (dimension == axis ?
                 op->ne[dimension] != op->src[0]->ne[dimension] + op->src[1]->ne[dimension] :
                 (op->ne[dimension] != op->src[0]->ne[dimension] || op->ne[dimension] != op->src[1]->ne[dimension]))) {
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
    if (n_dims <= 0 || n_dims > op->ne[0] || (uint32_t) n_dims / 2U > DAYAN_ROPE_MAX_PAIRS || (n_dims & 1) != 0 ||
        (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_MROPE &&
         mode != GGML_ROPE_TYPE_IMROPE)) {
        return false;
    }

    const int64_t position_axes = (mode & GGML_ROPE_TYPE_MROPE) != 0 ? 4 : 1;
    if (!ggml_is_vector(op->src[1]) || ggml_nelements(op->src[1]) != position_axes * op->ne[2]) {
        return false;
    }
    if (op->src[2] != nullptr && (op->src[2]->type != GGML_TYPE_F32 || ggml_nelements(op->src[2]) < n_dims / 2)) {
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
        (op->src[0]->type != GGML_TYPE_F32 && !ggml_backend_dayan_is_storage_type(op->src[0]->type)) ||
        !ggml_are_same_shape(op, op->src[0]) || op->ne[0] <= 0 ||
        op->ne[0] > UINT32_MAX || ggml_nrows(op) <= 0 || ggml_nrows(op) > UINT32_MAX || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous_rows(op->src[0])) {
        return false;
    }

    float epsilon;
    std::memcpy(&epsilon, op->op_params, sizeof(epsilon));
    return std::isfinite(epsilon) && epsilon >= 0.0f;
}

bool ggml_backend_dayan_supports_norm(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_NORM || op->src[0] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0]) || op->ne[0] <= 0 ||
        op->ne[0] > UINT32_MAX || ggml_nrows(op) <= 0 || ggml_nrows(op) > UINT32_MAX || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous_rows(op->src[0])) {
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

    const int32_t groups  = ggml_get_op_params_i32(op, 0);
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

bool ggml_backend_dayan_flash_attention_lengths(const ggml_tensor * op, uint32_t * causal_offset, uint32_t * kv_len) {
    if (op == nullptr || causal_offset == nullptr || kv_len == nullptr) {
        return false;
    }

    const ggml_tensor * mask = op->src[3];
    if (mask == nullptr) {
        if (op->src[1] == nullptr || op->src[1]->ne[1] <= 0 || op->src[1]->ne[1] > UINT32_MAX) {
            return false;
        }
        *causal_offset = 0;
        *kv_len        = (uint32_t) op->src[1]->ne[1];
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
            ggml_fp16_t  value_f16;
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
    *kv_len        = (uint32_t) effective_kv_len;
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
    std::memcpy(&logit_softcap, (const char *) op->op_params + sizeof(scale) + sizeof(max_bias), sizeof(logit_softcap));

    const int64_t       dimension_limit = std::numeric_limits<uint32_t>::max();
    const ggml_tensor * mask            = op->src[3];
    const bool          mask_supported =
        mask == nullptr || (mask->type == GGML_TYPE_F16 && mask->ne[0] == k->ne[1] && mask->ne[1] == q->ne[1] &&
                            mask->ne[2] == 1 && mask->ne[3] == 1 && ggml_is_contiguous_rows(mask));
    const bool input_type_supported =
        ggml_backend_dayan_is_storage_type(q->type) && k->type == q->type && v->type == q->type;
    return input_type_supported && ggml_is_contiguous_rows(q) && ggml_is_contiguous_rows(k) &&
           ggml_is_contiguous_rows(v) && q->ne[0] == k->ne[0] && q->ne[0] == v->ne[0] && q->ne[0] > 0 &&
           q->ne[0] <= 256 && q->ne[0] % 64 == 0 && q->ne[1] > 0 && q->ne[1] <= dimension_limit && k->ne[1] > 0 &&
           k->ne[1] <= dimension_limit && q->ne[2] > 0 && q->ne[2] <= dimension_limit && k->ne[2] > 0 &&
           k->ne[2] <= dimension_limit && q->ne[2] % k->ne[2] == 0 && q->ne[3] > 0 && q->ne[3] <= dimension_limit &&
           k->ne[3] == q->ne[3] && v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2] && v->ne[3] == k->ne[3] &&
           op->ne[0] == v->ne[0] && op->ne[1] == q->ne[2] && op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3] &&
           std::isfinite(scale) && scale >= 0.0f && max_bias == 0.0f && logit_softcap == 0.0f && mask_supported;
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
    if (op == nullptr || op->op != GGML_OP_CONT || op->src[0] == nullptr ||
        (op->type != GGML_TYPE_F32 && !ggml_backend_dayan_is_storage_type(op->type)) || !ggml_is_contiguous(op) ||
        op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }

    const ggml_tensor * view = op->src[0];
    const ggml_tensor * packed_qg = view->view_src != nullptr ? view->view_src : view->src[0];
    const ggml_tensor * qg = packed_qg != nullptr && ggml_backend_dayan_supports_quantized_mul_mat(packed_qg) ?
                                 packed_qg->src[0] :
                                 packed_qg;
    const size_t element_size = ggml_type_size(op->type);
    return view->op == GGML_OP_VIEW && packed_qg != nullptr && packed_qg->type == op->type && qg != nullptr &&
           ggml_backend_dayan_supports_mul_mat(qg) &&
           ggml_backend_dayan_is_storage_type(qg->src[0]->type) &&
           ggml_backend_dayan_mul_mat_has_single_batch(qg) &&
           view->type == op->type && view->ne[0] > 0 && view->ne[1] > 0 && view->ne[2] == qg->ne[1] &&
           view->ne[3] == 1 && op->ne[0] == view->ne[0] * view->ne[1] && op->ne[1] == view->ne[2] &&
           qg->ne[0] == 2 * op->ne[0] && view->nb[1] == 2 * view->ne[0] * element_size &&
           view->nb[2] == 2 * op->ne[0] * element_size && view->view_offs == view->ne[0] * element_size;
}

bool ggml_backend_dayan_supports_gate_sigmoid(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID &&
           op->src[0] != nullptr && op->type == op->src[0]->type && ggml_is_contiguous(op) &&
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
           weight->ne[0] <= kDayanSSMConvMaxKernelSize &&
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

struct ggml_backend_dayan_cast_input {
    ggml_tensor *               op;
    int                         source;
    ggml_backend_dayan_cast_key key;
};

struct ggml_backend_dayan_cast_hoist {
    ggml_backend_dayan_cast_key key;
    std::vector<ggml_tensor *>  layouts;
};

struct ggml_backend_dayan_gated_cast {
    ggml_backend_dayan_cast_key key;
    ggml_tensor *               projection;
    ggml_tensor *               qg;
    ggml_tensor *               gate_view;
    ggml_tensor *               gate_cont;
    ggml_tensor *               gate_sigmoid;
    ggml_tensor *               gated;
    ggml_tensor *               attention;
};

struct ggml_backend_dayan_residual_cast {
    ggml_tensor *               add;
    int                         source;
    ggml_backend_dayan_cast_key key;
};

struct ggml_backend_dayan_tensor_use {
    ggml_tensor * node;
    int           source;
};

using ggml_backend_dayan_use_map = std::unordered_map<ggml_tensor *, std::vector<ggml_backend_dayan_tensor_use>>;

static bool ggml_backend_dayan_is_cast_hoist_layout(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->src[0] == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    switch (tensor->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_dayan_layout_can_change_type(const ggml_tensor * tensor, enum ggml_type type) {
    const size_t source_size = ggml_type_size(tensor->type);
    const size_t target_size = ggml_type_size(type);
    if (source_size == 0 || target_size == 0 || source_size % target_size != 0 || tensor->buffer != nullptr ||
        tensor->data != nullptr || tensor->extra != nullptr || (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    const size_t scale = source_size / target_size;
    if (tensor->view_offs % scale != 0) {
        return false;
    }
    for (size_t stride : tensor->nb) {
        if (stride % scale != 0) {
            return false;
        }
    }
    if (tensor->op == GGML_OP_VIEW) {
        size_t offset;
        std::memcpy(&offset, tensor->op_params, sizeof(offset));
        if (offset % scale != 0) {
            return false;
        }
    }
    return true;
}

static void ggml_backend_dayan_change_layout_type(ggml_tensor * tensor, enum ggml_type type) {
    const size_t scale = ggml_type_size(tensor->type) / ggml_type_size(type);
    tensor->type       = type;
    tensor->view_offs /= scale;
    for (size_t & stride : tensor->nb) {
        stride /= scale;
    }
    if (tensor->op == GGML_OP_VIEW) {
        size_t offset;
        std::memcpy(&offset, tensor->op_params, sizeof(offset));
        offset /= scale;
        std::memcpy(tensor->op_params, &offset, sizeof(offset));
    }
}

static bool ggml_backend_dayan_has_single_use(const ggml_backend_dayan_use_map & uses,
                                              const ggml_tensor *                tensor,
                                              const ggml_tensor *                user,
                                              int                                source) {
    const auto found = uses.find(const_cast<ggml_tensor *>(tensor));
    return found != uses.end() && found->second.size() == 1 && found->second[0].node == user &&
           found->second[0].source == source;
}

static bool ggml_backend_dayan_norm_storage_type(const ggml_tensor *                rms,
                                                 const ggml_backend_dayan_use_map & uses,
                                                 enum ggml_type *                   type) {
    if (!ggml_backend_dayan_supports_rms_norm(rms) || type == nullptr) {
        return false;
    }

    const auto scale_use = uses.find(const_cast<ggml_tensor *>(rms));
    if (scale_use == uses.end() || scale_use->second.size() != 1 ||
        !ggml_backend_dayan_supports_rms_norm_scale(scale_use->second[0].node)) {
        return false;
    }

    const ggml_tensor * scale = scale_use->second[0].node;
    const auto          consumers = uses.find(const_cast<ggml_tensor *>(scale));
    if (consumers == uses.end() || consumers->second.empty()) {
        return false;
    }

    enum ggml_type storage_type = GGML_TYPE_COUNT;
    for (const ggml_backend_dayan_tensor_use & use : consumers->second) {
        if (use.source != 1 ||
            (!ggml_backend_dayan_supports_mul_mat(use.node) && !ggml_backend_dayan_supports_mul_mat_id(use.node))) {
            return false;
        }
        const enum ggml_type weight_type = use.node->src[0]->type;
        if (!ggml_backend_dayan_is_storage_type(weight_type) ||
            (storage_type != GGML_TYPE_COUNT && storage_type != weight_type)) {
            return false;
        }
        storage_type = weight_type;
    }
    *type = storage_type;
    return true;
}

static bool ggml_backend_dayan_activation_storage_type(const ggml_tensor *                activation,
                                                       const ggml_backend_dayan_use_map & uses,
                                                       enum ggml_type *                   type) {
    const auto consumers = uses.find(const_cast<ggml_tensor *>(activation));
    if (consumers == uses.end()) {
        return false;
    }
    for (const ggml_backend_dayan_tensor_use & use : consumers->second) {
        if (use.source == 0 && ggml_backend_dayan_norm_storage_type(use.node, uses, type)) {
            return true;
        }
    }
    return false;
}

using ggml_backend_dayan_residual_type_map = std::unordered_map<ggml_tensor *, enum ggml_type>;

static bool ggml_backend_dayan_residual_uses_are_compatible(
    const ggml_tensor *                         activation,
    enum ggml_type                              type,
    const ggml_backend_dayan_use_map &           uses,
    const ggml_backend_dayan_residual_type_map & residual_types) {
    const auto consumers = uses.find(const_cast<ggml_tensor *>(activation));
    if (consumers == uses.end() || consumers->second.empty()) {
        return false;
    }
    for (const ggml_backend_dayan_tensor_use & use : consumers->second) {
        enum ggml_type norm_type;
        if (use.source == 0 && ggml_backend_dayan_supports_rms_norm(use.node)) {
            if (!ggml_backend_dayan_norm_storage_type(use.node, uses, &norm_type) || norm_type == type) {
                continue;
            }
        }
        const auto residual = residual_types.find(use.node);
        if (use.source < 2 && residual != residual_types.end() && residual->second == type) {
            continue;
        }
        return false;
    }
    return true;
}

static bool ggml_backend_dayan_residual_source_storage_type(
    const ggml_tensor *                         input,
    const ggml_backend_dayan_residual_type_map & residual_types,
    enum ggml_type *                            type) {
    const auto residual = residual_types.find(const_cast<ggml_tensor *>(input));
    if (residual != residual_types.end()) {
        *type = residual->second;
        return true;
    }
    if (ggml_backend_dayan_is_storage_type(input->type)) {
        *type = input->type;
        return true;
    }
    if ((ggml_backend_dayan_supports_mul_mat(input) || ggml_backend_dayan_supports_mul_mat_id(input)) &&
        ggml_backend_dayan_is_storage_type(input->src[0]->type)) {
        *type = input->src[0]->type;
        return true;
    }
    return false;
}

static size_t ggml_backend_dayan_rewrite_get_rows_activation(
    ggml_tensor *                                    node,
    const ggml_backend_dayan_use_map &               uses,
    const ggml_backend_dayan_residual_type_map &     residual_types) {
    if (!ggml_backend_dayan_supports_get_rows(node) || node->type != GGML_TYPE_F32 ||
        (!ggml_backend_dayan_is_storage_type(node->src[0]->type) && node->src[0]->type != GGML_TYPE_MXFP4) ||
        node->view_src != nullptr || node->view_offs != 0) {
        return 0;
    }

    enum ggml_type type;
    if (!ggml_backend_dayan_activation_storage_type(node, uses, &type)) {
        return 0;
    }
    if ((node->src[0]->type != GGML_TYPE_MXFP4 && type != node->src[0]->type) ||
        !ggml_backend_dayan_layout_can_change_type(node, type) ||
        !ggml_backend_dayan_residual_uses_are_compatible(node, type, uses, residual_types)) {
        return 0;
    }

    ggml_backend_dayan_change_layout_type(node, type);
    return 1;
}

static bool ggml_backend_dayan_find_gated_cast(ggml_tensor *                      projection,
                                               const ggml_backend_dayan_use_map & uses,
                                               ggml_backend_dayan_gated_cast *    rewrite) {
    if (!ggml_backend_dayan_supports_mul_mat(projection) ||
        !ggml_backend_dayan_supports_gated_attention(projection->src[1])) {
        return false;
    }

    ggml_tensor * gated        = projection->src[1];
    const bool    left_gate    = ggml_backend_dayan_supports_gate_sigmoid(gated->src[0]);
    const int     gate_source  = left_gate ? 0 : 1;
    ggml_tensor * gate_sigmoid = gated->src[gate_source];
    ggml_tensor * attention    = gated->src[1 - gate_source];
    ggml_tensor * gate_cont    = gate_sigmoid->src[0];
    ggml_tensor * gate_view    = gate_cont->src[0];
    ggml_tensor * qg           = gate_view->view_src != nullptr ? gate_view->view_src : gate_view->src[0];
    const enum ggml_type type  = projection->src[0]->type;
    const bool view_can_change_type = ggml_backend_dayan_layout_can_change_type(gate_view, type);

    if (qg == nullptr || qg->src[0] == nullptr || qg->src[0]->type != type ||
        !view_can_change_type ||
        (qg->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || (gate_view->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        (gate_cont->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        (gate_sigmoid->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || (gated->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        !ggml_backend_dayan_has_single_use(uses, qg, gate_view, 0) ||
        !ggml_backend_dayan_has_single_use(uses, gate_view, gate_cont, 0) ||
        !ggml_backend_dayan_has_single_use(uses, gate_cont, gate_sigmoid, 0) ||
        !ggml_backend_dayan_has_single_use(uses, gate_sigmoid, gated, gate_source) ||
        !ggml_backend_dayan_has_single_use(uses, gated, projection, 1)) {
        return false;
    }

    *rewrite = { { qg, type }, projection, qg, gate_view, gate_cont, gate_sigmoid, gated, attention };
    return true;
}

static bool ggml_backend_dayan_find_cast_hoist(ggml_tensor *                      activation,
                                               enum ggml_type                     type,
                                               const ggml_backend_dayan_use_map & uses,
                                               ggml_backend_dayan_cast_hoist *    hoist) {
    if (type == GGML_TYPE_MXFP4) {
        return false;
    }
    std::vector<ggml_tensor *> layouts;
    ggml_tensor *              root = activation;
    while (ggml_backend_dayan_is_cast_hoist_layout(root)) {
        layouts.push_back(root);
        root = root->src[0];
    }
    if (layouts.empty() || !ggml_backend_dayan_supports_mul_mat(root) || root->type != GGML_TYPE_F32 ||
        (root->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    std::reverse(layouts.begin(), layouts.end());
    for (ggml_tensor * layout : layouts) {
        if (!ggml_backend_dayan_layout_can_change_type(layout, type)) {
            return false;
        }
    }

    ggml_tensor * expected = layouts.front();
    auto          found    = uses.find(root);
    if (found == uses.end() || found->second.size() != 1 || found->second[0].node != expected ||
        found->second[0].source != 0) {
        return false;
    }
    for (size_t index = 0; index + 1 < layouts.size(); ++index) {
        found = uses.find(layouts[index]);
        if (found == uses.end() || found->second.size() != 1 || found->second[0].node != layouts[index + 1] ||
            found->second[0].source != 0) {
            return false;
        }
    }

    found = uses.find(activation);
    if (found == uses.end() || found->second.empty()) {
        return false;
    }
    for (const ggml_backend_dayan_tensor_use & use : found->second) {
        if (use.source != 1 ||
            (!ggml_backend_dayan_supports_mul_mat(use.node) && !ggml_backend_dayan_supports_mul_mat_id(use.node)) ||
            use.node->src[0]->type != type) {
            return false;
        }
    }

    hoist->key     = { root, type };
    hoist->layouts = std::move(layouts);
    return true;
}

static ggml_tensor * ggml_backend_dayan_apply_cast_hoist(ggml_context *                        ctx,
                                                         const ggml_backend_dayan_cast_hoist & hoist,
                                                         ggml_tensor *                         cast) {
    ggml_tensor * source = cast;
    for (const ggml_tensor * layout : hoist.layouts) {
        const size_t scale  = ggml_type_size(layout->type) / ggml_type_size(hoist.key.type);
        size_t       offset = 0;
        if (layout->op == GGML_OP_VIEW) {
            std::memcpy(&offset, layout->op_params, sizeof(offset));
            offset /= scale;
        }

        ggml_tensor * replacement =
            ggml_view_4d(ctx, source, layout->ne[0], layout->ne[1], layout->ne[2], layout->ne[3], layout->nb[1] / scale,
                         layout->nb[2] / scale, layout->nb[3] / scale, offset);
        replacement->nb[0] = layout->nb[0] / scale;
        replacement->op    = layout->op;
        std::memcpy(replacement->op_params, layout->op_params, sizeof(replacement->op_params));
        if (replacement->op == GGML_OP_VIEW) {
            std::memcpy(replacement->op_params, &offset, sizeof(offset));
        }
        ggml_set_name(replacement, layout->name);
        source = replacement;
    }
    return source;
}

static void ggml_backend_dayan_apply_gated_cast(ggml_context *                       ctx,
                                                const ggml_backend_dayan_gated_cast & rewrite,
                                                ggml_tensor *                        cast,
                                                std::unordered_set<ggml_tensor *> *   elided_nodes) {
    ggml_backend_dayan_cast_hoist hoist = { rewrite.key, { rewrite.gate_view } };
    ggml_tensor * gate_view = ggml_backend_dayan_apply_cast_hoist(ctx, hoist, cast);
    ggml_tensor * gate_cont = ggml_cont_4d(ctx, gate_view, rewrite.gate_cont->ne[0], rewrite.gate_cont->ne[1],
                                          rewrite.gate_cont->ne[2], rewrite.gate_cont->ne[3]);
    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx, gate_cont);
    ggml_tensor * gated        = ggml_mul(ctx, rewrite.attention, gate_sigmoid);
    ggml_set_name(gate_cont, rewrite.gate_cont->name);
    ggml_set_name(gate_sigmoid, rewrite.gate_sigmoid->name);
    ggml_set_name(gated, rewrite.gated->name);
    rewrite.projection->src[1] = gated;

    elided_nodes->insert(rewrite.gate_view);
    elided_nodes->insert(rewrite.gate_cont);
    elided_nodes->insert(rewrite.gate_sigmoid);
    elided_nodes->insert(rewrite.gated);
}

static bool ggml_backend_dayan_flash_rewrite_type(const ggml_tensor * op, enum ggml_type * type) {
    if (op == nullptr || type == nullptr || op->op != GGML_OP_FLASH_ATTN_EXT || op->src[0] == nullptr ||
        op->src[1] == nullptr || op->src[2] == nullptr) {
        return false;
    }
    for (int source = 0; source < 3; ++source) {
        if (!ggml_backend_dayan_is_flash_type(op->src[source]->type)) {
            return false;
        }
    }

    const enum ggml_type k_type = op->src[1]->type;
    const enum ggml_type v_type = op->src[2]->type;
    if (ggml_backend_dayan_is_storage_type(k_type) && ggml_backend_dayan_is_storage_type(v_type) && k_type != v_type) {
        return false;
    }
    if (ggml_backend_dayan_is_storage_type(k_type)) {
        *type = k_type;
    } else if (ggml_backend_dayan_is_storage_type(v_type)) {
        *type = v_type;
    } else if (ggml_backend_dayan_is_storage_type(op->src[0]->type)) {
        *type = op->src[0]->type;
    } else {
        *type = GGML_TYPE_BF16;
    }
    return true;
}

static int ggml_backend_dayan_next_compute(const ggml_cgraph * cgraph, int index);

static bool ggml_backend_dayan_add_dependency(ggml_tensor * node, ggml_tensor * dependency) {
    for (ggml_tensor * source : node->src) {
        if (source == dependency) {
            return true;
        }
    }
    for (int index = 2; index < GGML_MAX_SRC; ++index) {
        if (node->src[index] == nullptr) {
            node->src[index] = dependency;
            return true;
        }
    }
    return false;
}

static bool ggml_backend_dayan_preserve_norm_pipeline_inputs(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes;) {
        i = ggml_backend_dayan_next_compute(cgraph, i);
        if (i >= cgraph->n_nodes) {
            break;
        }

        ggml_tensor * add         = nullptr;
        ggml_tensor * rms         = nullptr;
        int           scale_index = -1;
        if (ggml_backend_dayan_supports_residual_add(cgraph->nodes[i])) {
            add                 = cgraph->nodes[i];
            const int rms_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
            scale_index         = ggml_backend_dayan_next_compute(cgraph, rms_index + 1);
            if (rms_index < cgraph->n_nodes && scale_index < cgraph->n_nodes &&
                ggml_backend_dayan_supports_rms_norm(cgraph->nodes[rms_index]) &&
                cgraph->nodes[rms_index]->src[0] == add) {
                rms = cgraph->nodes[rms_index];
            }
        } else if (ggml_backend_dayan_supports_rms_norm(cgraph->nodes[i])) {
            rms         = cgraph->nodes[i];
            scale_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
        }

        if (rms == nullptr || scale_index < 0 || scale_index >= cgraph->n_nodes ||
            !ggml_backend_dayan_supports_rms_norm_scale(cgraph->nodes[scale_index]) ||
            (cgraph->nodes[scale_index]->src[0] != rms && cgraph->nodes[scale_index]->src[1] != rms)) {
            ++i;
            continue;
        }

        ggml_tensor * scale_op    = cgraph->nodes[scale_index];
        ggml_tensor * final_op    = scale_op;
        int           final_index = scale_index;
        const int     cast_index  = ggml_backend_dayan_next_compute(cgraph, scale_index + 1);
        if (cast_index < cgraph->n_nodes && ggml_backend_dayan_supports_quantized_rms_norm(cgraph->nodes[cast_index]) &&
            cgraph->nodes[cast_index]->src[0] == scale_op) {
            final_op    = cgraph->nodes[cast_index];
            final_index = cast_index;
        }

        ggml_tensor * scale = scale_op->src[0] == rms ? scale_op->src[1] : scale_op->src[0];
        // Keep fused inputs live through the last logical operation.
        if ((add == nullptr && !ggml_backend_dayan_add_dependency(final_op, rms->src[0])) ||
            (add != nullptr && (!ggml_backend_dayan_add_dependency(final_op, add->src[0]) ||
                                !ggml_backend_dayan_add_dependency(final_op, add->src[1]))) ||
            !ggml_backend_dayan_add_dependency(final_op, scale)) {
            return false;
        }
        i = final_index + 1;
    }
    return true;
}

static bool ggml_backend_dayan_preserve_matmul_inputs(const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * output = graph->nodes[i];
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        if (ggml_backend_dayan_supports_quantized_mul_mat(output) || ggml_backend_dayan_supports_mul_mat_silu(output)) {
            gate = output->src[0];
        } else if (ggml_backend_dayan_supports_quantized_glu(output)) {
            gate = output->src[0]->src[0];
            up = output->src[0]->src[1];
        }
        if (gate == nullptr) {
            continue;
        }
        // Keep inputs live until the fused output, including payload and scale planes.
        if (!ggml_backend_dayan_add_dependency(output, gate->src[0]) ||
            !ggml_backend_dayan_add_dependency(output, gate->src[1]) ||
            (up != nullptr && !ggml_backend_dayan_add_dependency(output, up->src[0]))) {
            return false;
        }
        if (gate->op == GGML_OP_MUL_MAT_ID && !ggml_backend_dayan_add_dependency(output, gate->src[2])) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_dayan_preserve_ssm_conv_inputs(const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * output = graph->nodes[i];
        ggml_tensor * conv = ggml_backend_dayan_supports_ssm_conv_silu(output) ? output->src[0] : output;
        if (!ggml_backend_dayan_supports_ssm_conv(conv) || conv->src[0]->op != GGML_OP_CONCAT) {
            continue;
        }
        ggml_tensor * concat = conv->src[0];
        if (!ggml_backend_dayan_add_dependency(output, concat->src[0]) ||
            !ggml_backend_dayan_add_dependency(output, concat->src[1]) ||
            !ggml_backend_dayan_add_dependency(output, conv->src[1])) {
            return false;
        }
        for (int j = 0; j < i; ++j) {
            ggml_tensor * copy = graph->nodes[j];
            if (copy->op == GGML_OP_CPY && copy->src[0]->view_src == concat &&
                !ggml_backend_dayan_add_dependency(output, copy)) {
                return false;
            }
        }
    }
    return true;
}

bool ggml_backend_dayan_graph_rewrite(ggml_backend_t backend, ggml_context * ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    if (ctx == nullptr || cgraph == nullptr) {
        return false;
    }

    const int64_t rewrite_start_us = ggml_time_us();

    std::vector<ggml_tensor *> nodes(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    std::vector<std::pair<ggml_tensor *, ggml_backend_dayan_cast_key>>                               matmuls;
    std::vector<ggml_backend_dayan_cast_input>                                                       flash_inputs;
    std::vector<ggml_backend_dayan_residual_cast>                                                    residual_inputs;
    std::vector<ggml_backend_dayan_gated_cast>                                                       gated_casts;
    std::vector<ggml_backend_dayan_cast_key>                                                         cast_keys;
    std::unordered_map<ggml_backend_dayan_cast_key, ggml_tensor *, ggml_backend_dayan_cast_key_hash> casts;
    ggml_backend_dayan_use_map                                                                       uses;
    std::unordered_map<ggml_tensor *, enum ggml_type>                                                activation_types;
    std::unordered_set<ggml_tensor *>                                conflicting_activations;
    std::unordered_map<ggml_tensor *, ggml_backend_dayan_cast_hoist> hoists;
    std::unordered_map<ggml_tensor *, ggml_tensor *>                 hoisted_replacements;
    std::unordered_set<ggml_tensor *>                                elided_layouts;
    size_t                                                           gated_f32 = 0;
    size_t                                                           embedding_lowp = 0;
    size_t                                                           residual_lowp = 0;
    bool                                                             has_moe = false;

    for (ggml_tensor * node : nodes) {
        has_moe = has_moe || node->op == GGML_OP_MUL_MAT_ID;
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            if (node->src[source] != nullptr) {
                uses[node->src[source]].push_back({ node, source });
            }
        }
    }

    ggml_backend_dayan_residual_type_map residual_types;
    // Keep MoE residuals in F32 to avoid changing close router scores.
    for (ggml_tensor * node : nodes) {
        enum ggml_type type;
        if (!has_moe && ggml_backend_dayan_supports_residual_add(node) &&
            ggml_backend_dayan_activation_storage_type(node, uses, &type) &&
            ggml_backend_dayan_layout_can_change_type(node, type)) {
            residual_types.emplace(node, type);
        }
    }
    for (bool changed = !has_moe; changed;) {
        changed = false;
        for (ggml_tensor * node : nodes) {
            if (!ggml_backend_dayan_supports_residual_add(node) || residual_types.count(node) != 0) {
                continue;
            }
            enum ggml_type type = GGML_TYPE_COUNT;
            for (int source = 0; source < 2; ++source) {
                enum ggml_type source_type;
                if (ggml_backend_dayan_residual_source_storage_type(
                        node->src[source], residual_types, &source_type)) {
                    if (type != GGML_TYPE_COUNT && type != source_type) {
                        type = GGML_TYPE_COUNT;
                        break;
                    }
                    type = source_type;
                }
            }
            if (ggml_backend_dayan_is_storage_type(type) && ggml_backend_dayan_layout_can_change_type(node, type)) {
                residual_types.emplace(node, type);
                changed = true;
            }
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (auto residual = residual_types.begin(); residual != residual_types.end();) {
            if (!ggml_backend_dayan_residual_uses_are_compatible(
                    residual->first, residual->second, uses, residual_types)) {
                residual = residual_types.erase(residual);
                changed  = true;
            } else {
                ++residual;
            }
        }
    }
    for (ggml_tensor * node : nodes) {
        embedding_lowp += ggml_backend_dayan_rewrite_get_rows_activation(node, uses, residual_types);
    }
    for (const auto & residual : residual_types) {
        ggml_tensor *       add  = residual.first;
        const enum ggml_type type = residual.second;
        ggml_backend_dayan_change_layout_type(add, type);
        ++residual_lowp;
    }
    for (const auto & residual : residual_types) {
        ggml_tensor * add  = residual.first;
        const enum ggml_type type = residual.second;
        if (add->type != type) {
            continue;
        }
        for (int source = 0; source < 2; ++source) {
            if (add->src[source]->type == type) {
                continue;
            }
            const ggml_backend_dayan_cast_key key = { add->src[source], type };
            residual_inputs.push_back({ add, source, key });
            if (casts.emplace(key, nullptr).second) {
                cast_keys.push_back(key);
            }
        }
    }
    for (ggml_tensor * node : nodes) {
        if ((ggml_backend_dayan_supports_mul_mat(node) || ggml_backend_dayan_supports_mul_mat_id(node)) &&
            node->src[0]->type != GGML_TYPE_F32 &&
            (node->src[1]->type == GGML_TYPE_F32 || (node->src[0]->type == GGML_TYPE_MXFP4 && node->src[1]->type != GGML_TYPE_MXFP4)) &&
            (node->src[0]->type == GGML_TYPE_MXFP4 || !ggml_backend_dayan_supports_gated_attention(node->src[1]))) {
            auto inserted = activation_types.emplace(node->src[1], node->src[0]->type);
            if (!inserted.second && inserted.first->second != node->src[0]->type) {
                conflicting_activations.insert(node->src[1]);
            }
        }
    }
    for (const auto & activation : activation_types) {
        if (conflicting_activations.count(activation.first) == 0) {
            ggml_backend_dayan_cast_hoist hoist;
            if (ggml_backend_dayan_find_cast_hoist(activation.first, activation.second, uses, &hoist)) {
                hoists.emplace(activation.first, std::move(hoist));
            }
        }
    }

    for (ggml_tensor * node : nodes) {
        if ((ggml_backend_dayan_supports_mul_mat(node) || ggml_backend_dayan_supports_mul_mat_id(node)) &&
            node->src[0]->type != GGML_TYPE_F32 &&
            (node->src[1]->type == GGML_TYPE_F32 || (node->src[0]->type == GGML_TYPE_MXFP4 && node->src[1]->type != GGML_TYPE_MXFP4))) {
            if (node->src[0]->type != GGML_TYPE_MXFP4 && ggml_backend_dayan_supports_gated_attention(node->src[1])) {
                ggml_backend_dayan_gated_cast rewrite;
                if (ggml_backend_dayan_find_gated_cast(node, uses, &rewrite)) {
                    gated_casts.push_back(rewrite);
                    if (casts.emplace(rewrite.key, nullptr).second) {
                        cast_keys.push_back(rewrite.key);
                    }
                } else {
                    ++gated_f32;
                }
                continue;
            }
            const auto                        hoist = hoists.find(node->src[1]);
            const ggml_backend_dayan_cast_key key =
                hoist == hoists.end() ? ggml_backend_dayan_cast_key{ node->src[1], node->src[0]->type } :
                                        hoist->second.key;
            matmuls.emplace_back(node, key);
            if (casts.emplace(key, nullptr).second) {
                cast_keys.push_back(key);
            }
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            enum ggml_type target_type;
            if (!ggml_backend_dayan_flash_rewrite_type(node, &target_type)) {
                continue;
            }
            for (int source = 0; source < 3; ++source) {
                ggml_tensor * input = node->src[source];
                if (input->type == target_type) {
                    continue;
                }
                const ggml_backend_dayan_cast_key key = { input, target_type };
                flash_inputs.push_back({ node, source, key });
                if (casts.emplace(key, nullptr).second) {
                    cast_keys.push_back(key);
                }
            }
        }
    }

    if (cast_keys.empty()) {
        const bool success = ggml_backend_dayan_preserve_norm_pipeline_inputs(cgraph) &&
                             ggml_backend_dayan_preserve_matmul_inputs(cgraph) &&
                             ggml_backend_dayan_preserve_ssm_conv_inputs(cgraph);
        GGML_LOG_DEBUG("%s: no activation casts, embedding-lowp=%zu, residual-lowp=%zu, gated-f32=%zu, time=%.3f ms\n",
                       __func__, embedding_lowp, residual_lowp, gated_f32,
                       (double) (ggml_time_us() - rewrite_start_us) / 1000.0);
        return success;
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

    for (const auto & hoist : hoists) {
        ggml_tensor * replacement = ggml_backend_dayan_apply_cast_hoist(ctx, hoist.second, casts.at(hoist.second.key));
        hoisted_replacements.emplace(hoist.first, replacement);
        elided_layouts.insert(hoist.second.layouts.begin(), hoist.second.layouts.end());
    }
    for (const ggml_backend_dayan_gated_cast & rewrite : gated_casts) {
        ggml_backend_dayan_apply_gated_cast(ctx, rewrite, casts.at(rewrite.key), &elided_layouts);
    }

    for (const auto & matmul : matmuls) {
        const auto replacement = hoisted_replacements.find(matmul.first->src[1]);
        if (replacement == hoisted_replacements.end()) {
            matmul.first->src[1] = casts.at(matmul.second);
        } else {
            matmul.first->src[1] = replacement->second;
        }
    }
    for (const ggml_backend_dayan_residual_cast & input : residual_inputs) {
        input.add->src[input.source] = casts.at(input.key);
    }
    for (const ggml_backend_dayan_cast_input & input : flash_inputs) {
        input.op->src[input.source] = casts.at(input.key);
    }

    ggml_graph_clear(cgraph);
    for (ggml_tensor * node : nodes) {
        if (elided_layouts.count(node) == 0) {
            ggml_build_forward_expand(cgraph, node);
        }
    }
    if (!ggml_backend_dayan_preserve_norm_pipeline_inputs(cgraph) || !ggml_backend_dayan_preserve_matmul_inputs(cgraph) ||
        !ggml_backend_dayan_preserve_ssm_conv_inputs(cgraph)) {
        GGML_LOG_WARN("%s: graph has no source slots for fused input dependencies\n", __func__);
        return false;
    }

    GGML_LOG_DEBUG(
        "%s: rewrote %zu embedding paths, %zu residual paths, %zu matrix multiplications, %zu gated QG paths, kept %zu gated QG paths "
        "in F32, and %zu "
        "FlashAttention inputs with %zu "
        "low-precision casts, hoisted=%zu, time=%.3f ms\n",
        __func__, embedding_lowp, residual_lowp, matmuls.size(), gated_casts.size(), gated_f32, flash_inputs.size(),
        cast_keys.size(), hoists.size(), (double) (ggml_time_us() - rewrite_start_us) / 1000.0);
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
                                       ggml_is_empty(cgraph->nodes[index]) ||
                                       ggml_backend_dayan_is_metadata(cgraph->nodes[index]))) {
        ++index;
    }
    return index;
}

static bool ggml_backend_dayan_can_elide_glu_intermediates(const ggml_cgraph * cgraph,
                                                           int                 first_index,
                                                           int                 second_index,
                                                           int                 glu_index) {
    return first_index >= 0 && second_index >= 0 && glu_index >= 0 && first_index < cgraph->n_nodes &&
           second_index < cgraph->n_nodes && glu_index < cgraph->n_nodes &&
           ggml_node_has_n_uses(cgraph, first_index, 1) && ggml_node_has_n_uses(cgraph, second_index, 1) &&
           ggml_node_has_n_uses(cgraph, glu_index, 1);
}

static bool ggml_backend_dayan_mxfp4_inputs_live(const ggml_tensor * output, const ggml_tensor * first,
                                                const ggml_tensor * second = nullptr) {
    if (output->type != GGML_TYPE_MXFP4 && first->src[0]->type != GGML_TYPE_MXFP4) {
        return true;
    }
    const ggml_tensor * inputs[] = {first->src[0], first->src[1], second ? second->src[0] : first->src[0],
                                   first->op == GGML_OP_MUL_MAT_ID ? first->src[2] : first->src[0]};
    for (const ggml_tensor * input : inputs) {
        bool found = false;
        for (const ggml_tensor * source : output->src) { found |= source == input; }
        if (!found) { return false; }
    }
    return true;
}

static void ggml_backend_dayan_set_norm_pipeline_outputs(const ggml_cgraph *             cgraph,
                                                         ggml_backend_dayan_graph_slot * slot,
                                                         int                             add_index,
                                                         int                             rms_index,
                                                         int                             scale_index,
                                                         int                             cast_index) {
    if (add_index >= 0 && !ggml_node_has_n_uses(cgraph, add_index, 1)) {
        slot->output_mask |= kDayanNormSlotOutputResidual;
    }
    if (!ggml_node_has_n_uses(cgraph, rms_index, 1)) {
        slot->output_mask |= kDayanNormSlotOutputRms;
    }
    if (cast_index >= 0) {
        if (!ggml_node_has_n_uses(cgraph, scale_index, 1)) {
            slot->output_mask |= kDayanNormSlotOutputScaled;
        }
        slot->output_mask |= kDayanNormSlotOutputLowp;
    } else {
        slot->output_mask |= kDayanNormSlotOutputScaled;
    }
}

static void ggml_backend_dayan_fuse_ssm_conv(const ggml_cgraph * graph, ggml_backend_dayan_graph_plan * plan) {
    std::unordered_set<const ggml_tensor *> elided;
    for (auto & slot : plan->slots) {
        if (slot.kind != ggml_backend_dayan_slot_kind::ssm_conv && slot.kind != ggml_backend_dayan_slot_kind::ssm_conv_silu) {
            continue;
        }
        ggml_tensor * conv = slot.nodes[0];
        ggml_tensor * concat = conv->src[0];
        if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0 ||
            concat->src[1]->op != GGML_OP_TRANSPOSE || (concat->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            continue;
        }
        ggml_tensor * state = concat->src[0];
        ggml_tensor * qkv = concat->src[1]->src[0];
        const int64_t history = conv->src[1]->ne[0] - 1;
        if (history < 1 || state->type != GGML_TYPE_F32 || qkv->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(state) || !ggml_is_contiguous(qkv) || state->ne[0] != history ||
            state->ne[1] != qkv->ne[0] || state->ne[2] != qkv->ne[2] || state->ne[3] != 1 || qkv->ne[3] != 1) {
            continue;
        }
        ggml_tensor * copy = nullptr;
        bool found_concat = false;
        for (const auto & candidate : plan->slots) {
            if (candidate.nodes[0] == conv) {
                break;
            }
            found_concat |= candidate.nodes[0] == concat;
            ggml_tensor * node = candidate.nodes[0];
            if (candidate.kind == ggml_backend_dayan_slot_kind::copy && node->src[0]->view_src == concat) {
                if (copy != nullptr) {
                    copy = nullptr;
                    break;
                }
                copy = node;
            }
        }
        if (!found_concat || copy == nullptr || copy->src[0]->op != GGML_OP_VIEW ||
            copy->src[0]->view_offs != (size_t) qkv->ne[1] * sizeof(float) ||
            copy->type != GGML_TYPE_F32 || !ggml_is_contiguous(copy) ||
            copy->src[0]->ne[0] != history || copy->src[0]->ne[1] != state->ne[1] ||
            copy->src[0]->ne[2] != state->ne[2] || ggml_nelements(copy) != ggml_nelements(state) ||
            copy->src[0]->nb[0] != concat->nb[0] || copy->src[0]->nb[1] != concat->nb[1] ||
            copy->src[0]->nb[2] != concat->nb[2] ||
            (copy->src[0]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            continue;
        }
        const ggml_tensor * output = slot.nodes[1] != nullptr ? slot.nodes[1] : conv;
        bool inputs_live = true;
        for (const ggml_tensor * input : { state, concat->src[1], conv->src[1], copy }) {
            bool found = false;
            for (const ggml_tensor * source : output->src) { found |= source == input; }
            inputs_live &= found;
        }
        if (!inputs_live) {
            continue;
        }
        const ggml_tensor * destination = copy->view_src != nullptr ? copy->view_src : copy;
        const ggml_tensor * state_root = state->view_src != nullptr ? state->view_src : state;
        const ggml_tensor * qkv_root = qkv->view_src != nullptr ? qkv->view_src : qkv;
        bool compatible = destination != state_root && destination != qkv_root;
        bool before_conv = false;
        for (int i = 0; i < graph->n_nodes; ++i) {
            const ggml_tensor * node = graph->nodes[i];
            if (node == conv) { before_conv = false; }
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                const ggml_tensor * input = node->src[source];
                if ((input == concat && node != conv && node != copy->src[0]) ||
                    (input == copy->src[0] && node != copy)) {
                    compatible = false;
                }
                if (before_conv && !ggml_backend_dayan_supports_layout(node) && input != nullptr &&
                    (input->view_src != nullptr ? input->view_src : input) == destination) {
                    compatible = false;
                }
            }
            if (node == copy) { before_conv = true; }
        }
        if (!compatible) {
            continue;
        }
        slot.nodes[2] = concat;
        slot.nodes[3] = copy;
        slot.node_count = 4;
        elided.insert(concat);
        elided.insert(copy);
    }
    plan->slots.erase(std::remove_if(plan->slots.begin(), plan->slots.end(),
        [&elided](const ggml_backend_dayan_graph_slot & slot) { return elided.count(slot.nodes[0]) != 0; }), plan->slots.end());
}

bool ggml_backend_dayan_compile_graph(const ggml_cgraph *             cgraph,
                                      ggml_backend_dayan_graph_plan * plan,
                                      const ggml_tensor **            unsupported) {
    plan->slots.clear();
    plan->paired_mul_mat_slots  = 0;
    plan->fused_mul_mat_slots   = 0;
    plan->residual_add_slots    = 0;
    plan->rms_norm_slots        = 0;
    plan->flash_attention_slots = 0;
    plan->gated_delta_net_slots = 0;
    plan->gated_output_slots    = 0;
    plan->ssm_conv_slots        = 0;
    plan->moe_slots             = 0;
    plan->fused_moe_slots       = 0;
    plan->routing_slots         = 0;
    *unsupported                = nullptr;

    std::unordered_map<const ggml_tensor *, int>           node_indices;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> quantized_outputs;
    std::unordered_set<const ggml_tensor *>                elided_nodes;
    for (int index = 0; index < cgraph->n_nodes; ++index) {
        node_indices.emplace(cgraph->nodes[index], index);
    }
    for (int index = 0; index < cgraph->n_nodes; ++index) {
        ggml_tensor * output = cgraph->nodes[index];
        if (!ggml_backend_dayan_supports_quantized_mul_mat(output) ||
            !ggml_backend_dayan_mxfp4_inputs_live(output, output->src[0])) {
            continue;
        }
        const auto producer = node_indices.find(output->src[0]);
        if (producer != node_indices.end() && producer->second < index &&
            ggml_node_has_n_uses(cgraph, producer->second, 1)) {
            quantized_outputs.emplace(output->src[0], output);
        }
    }

    for (int i = 0; i < cgraph->n_nodes;) {
        i = ggml_backend_dayan_next_compute(cgraph, i);
        if (i >= cgraph->n_nodes) {
            break;
        }

        ggml_tensor * node = cgraph->nodes[i];
        if (elided_nodes.count(node) != 0) {
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_residual_add(node)) {
            const int                     add_index = i;
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
                slot.kind       = ggml_backend_dayan_slot_kind::norm_pipeline;
                slot.nodes[0]   = node;
                slot.nodes[1]   = cgraph->nodes[rms_index];
                slot.nodes[2]   = cgraph->nodes[scale_index];
                slot.node_count = 3;
                i               = scale_index + 1;

                int cast_index = ggml_backend_dayan_next_compute(cgraph, i);
                if (cast_index < cgraph->n_nodes &&
                    ggml_backend_dayan_supports_quantized_rms_norm(cgraph->nodes[cast_index]) &&
                    cgraph->nodes[cast_index]->src[0] == cgraph->nodes[scale_index]) {
                    slot.nodes[3]   = cgraph->nodes[cast_index];
                    slot.node_count = 4;
                    i               = cast_index + 1;
                } else {
                    cast_index = -1;
                }
                ggml_backend_dayan_set_norm_pipeline_outputs(cgraph, &slot, add_index, rms_index, scale_index,
                                                             cast_index);
                ++plan->rms_norm_slots;
            } else {
                ++i;
            }
            plan->slots.push_back(slot);
            ++plan->residual_add_slots;
            continue;
        }
        if (ggml_backend_dayan_supports_rms_norm(node)) {
            const int                     rms_index = i;
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::rms_norm;
            slot.nodes[0]   = node;
            slot.node_count = 1;

            const int scale_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
            if (scale_index < cgraph->n_nodes &&
                ggml_backend_dayan_supports_rms_norm_scale(cgraph->nodes[scale_index]) &&
                (cgraph->nodes[scale_index]->src[0] == node || cgraph->nodes[scale_index]->src[1] == node)) {
                slot.kind       = ggml_backend_dayan_slot_kind::norm_pipeline;
                slot.nodes[0]   = nullptr;
                slot.nodes[1]   = node;
                slot.nodes[2]   = cgraph->nodes[scale_index];
                slot.node_count = 2;
                i               = scale_index + 1;

                int cast_index = ggml_backend_dayan_next_compute(cgraph, i);
                if (cast_index < cgraph->n_nodes &&
                    ggml_backend_dayan_supports_quantized_rms_norm(cgraph->nodes[cast_index]) &&
                    cgraph->nodes[cast_index]->src[0] == cgraph->nodes[scale_index]) {
                    slot.nodes[3]   = cgraph->nodes[cast_index];
                    slot.node_count = 3;
                    i               = cast_index + 1;
                } else {
                    cast_index = -1;
                }
                ggml_backend_dayan_set_norm_pipeline_outputs(cgraph, &slot, -1, rms_index, scale_index, cast_index);
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
                cgraph->nodes[silu_index]->src[0] == node && ggml_node_has_n_uses(cgraph, i, 1)) {
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
            ggml_tensor *                 projection = cgraph->nodes[projection_index];
            if (gate_view->ne[0] % 64 != 0 || gate_view->ne[0] > 256 || 256 % gate_view->ne[0] != 0 ||
                projection->ne[1] >= projection->ne[0]) {
                *unsupported = node;
                return false;
            }
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::gated_output_projection;
            slot.nodes[0]   = node;
            slot.nodes[1]   = cgraph->nodes[mul_index];
            slot.nodes[2]   = projection;
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
        if (ggml_backend_dayan_supports_add_id(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::add_id;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            plan->slots.push_back(slot);
            ++plan->moe_slots;
            ++i;
            continue;
        }
        if (ggml_backend_dayan_supports_argsort(node) || ggml_backend_dayan_supports_top_k(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = node->op == GGML_OP_ARGSORT ? ggml_backend_dayan_slot_kind::argsort :
                                                            ggml_backend_dayan_slot_kind::top_k;
            slot.nodes[0]   = node;
            slot.node_count = 1;
            slot.sort_k     = (uint32_t) node->ne[0];
            if (node->op == GGML_OP_ARGSORT && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                uint32_t prefix = 0;
                for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                    const ggml_tensor * user = cgraph->nodes[j];
                    for (int source = 0; source < GGML_MAX_SRC; ++source) {
                        if (user->src[source] != node) {
                            continue;
                        }
                        if (user->op != GGML_OP_VIEW || source != 0 || user->view_offs != 0 ||
                            user->nb[0] != node->nb[0] || user->nb[1] != node->nb[1] ||
                            user->nb[2] != node->nb[2] || user->nb[3] != node->nb[3] ||
                            user->ne[1] != node->ne[1] || user->ne[2] != node->ne[2] ||
                            user->ne[3] != node->ne[3]) {
                            prefix = (uint32_t) node->ne[0];
                        } else {
                            prefix = std::max(prefix, (uint32_t) user->ne[0]);
                        }
                    }
                }
                if (prefix > 0) {
                    slot.sort_k = prefix;
                }
            }
            plan->slots.push_back(slot);
            ++plan->routing_slots;
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
        if (ggml_backend_dayan_supports_mul_mat_id(node)) {
            ggml_backend_dayan_graph_slot slot;
            slot.kind       = ggml_backend_dayan_slot_kind::mul_mat_id;
            slot.nodes[0]   = node;
            slot.node_count = 1;

            const int pair_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
            if (pair_index < cgraph->n_nodes && ggml_backend_dayan_supports_mul_mat_id(cgraph->nodes[pair_index]) &&
                node->src[1] == cgraph->nodes[pair_index]->src[1] &&
                node->src[2] == cgraph->nodes[pair_index]->src[2] &&
                node->src[0]->type == cgraph->nodes[pair_index]->src[0]->type) {
                ggml_tensor * pair       = cgraph->nodes[pair_index];
                const int     glu_index  = ggml_backend_dayan_next_compute(cgraph, pair_index + 1);
                const int     cast_index = ggml_backend_dayan_next_compute(cgraph, glu_index + 1);
                if (glu_index < cgraph->n_nodes && cast_index < cgraph->n_nodes &&
                    ggml_backend_dayan_can_elide_glu_intermediates(cgraph, i, pair_index, glu_index) &&
                    ggml_backend_dayan_supports_mul_mat_id_glu(cgraph->nodes[glu_index]) &&
                    ggml_backend_dayan_supports_quantized_glu(cgraph->nodes[cast_index]) &&
                    ggml_backend_dayan_mxfp4_inputs_live(cgraph->nodes[cast_index], node, pair) &&
                    ((cgraph->nodes[glu_index]->src[0] == node && cgraph->nodes[glu_index]->src[1] == pair) ||
                     (cgraph->nodes[glu_index]->src[0] == pair && cgraph->nodes[glu_index]->src[1] == node)) &&
                    cgraph->nodes[cast_index]->src[0] == cgraph->nodes[glu_index]) {
                    ggml_tensor * glu = cgraph->nodes[glu_index];
                    slot.kind         = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU ?
                                            ggml_backend_dayan_slot_kind::mul_mat_id_swiglu :
                                            ggml_backend_dayan_slot_kind::mul_mat_id_geglu;
                    slot.nodes[0]     = glu->src[0];
                    slot.nodes[1]     = glu->src[1];
                    slot.nodes[2]     = glu;
                    slot.nodes[3]     = cgraph->nodes[cast_index];
                    slot.node_count   = 4;
                    i                 = cast_index + 1;
                    ++plan->fused_moe_slots;
                } else {
                    slot.kind       = ggml_backend_dayan_slot_kind::mul_mat_id_pair;
                    slot.nodes[1]   = pair;
                    slot.node_count = 2;
                    i               = pair_index + 1;
                }
            } else {
                ++i;
            }
            plan->slots.push_back(slot);
            ++plan->moe_slots;
            continue;
        }
        if (!ggml_backend_dayan_supports_mul_mat(node)) {
            *unsupported = node;
            return false;
        }

        ggml_backend_dayan_graph_slot slot;
        slot.nodes[0]   = node;
        slot.node_count = 1;

        const auto quantized = quantized_outputs.find(node);
        if (node->src[0]->type == GGML_TYPE_F32) {
            slot.kind = ggml_backend_dayan_slot_kind::mul_mat;
            ++i;
            plan->slots.push_back(slot);
            continue;
        }
        if (quantized != quantized_outputs.end()) {
            slot.kind       = ggml_backend_dayan_slot_kind::mul_mat_quantized;
            slot.nodes[1]   = quantized->second;
            slot.node_count = 2;
            ++plan->fused_mul_mat_slots;
            elided_nodes.insert(quantized->second);
            ++i;
            plan->slots.push_back(slot);
            continue;
        }

        const int silu_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
        if (silu_index < cgraph->n_nodes && ggml_backend_dayan_supports_mul_mat_silu(cgraph->nodes[silu_index]) &&
            ggml_backend_dayan_mxfp4_inputs_live(cgraph->nodes[silu_index], node) &&
            cgraph->nodes[silu_index]->src[0] == node && ggml_node_has_n_uses(cgraph, i, 1) &&
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            slot.kind       = ggml_backend_dayan_slot_kind::mul_mat_silu;
            slot.nodes[1]   = cgraph->nodes[silu_index];
            slot.node_count = 2;
            ++plan->fused_mul_mat_slots;
            i = silu_index + 1;
            plan->slots.push_back(slot);
            continue;
        }

        const int pair_index = ggml_backend_dayan_next_compute(cgraph, i + 1);
        if (ggml_backend_dayan_mul_mat_has_single_batch(node) && pair_index < cgraph->n_nodes &&
            ggml_backend_dayan_supports_mul_mat(cgraph->nodes[pair_index]) &&
            ggml_backend_dayan_mul_mat_has_single_batch(cgraph->nodes[pair_index]) &&
            node->src[1] == cgraph->nodes[pair_index]->src[1] &&
            node->src[0]->type == cgraph->nodes[pair_index]->src[0]->type) {
            ggml_tensor * pair       = cgraph->nodes[pair_index];
            const int     glu_index  = ggml_backend_dayan_next_compute(cgraph, pair_index + 1);
            const int     cast_index = ggml_backend_dayan_next_compute(cgraph, glu_index + 1);

            if (glu_index < cgraph->n_nodes && cast_index < cgraph->n_nodes &&
                ggml_backend_dayan_can_elide_glu_intermediates(cgraph, i, pair_index, glu_index) &&
                ggml_backend_dayan_supports_glu(cgraph->nodes[glu_index]) &&
                ggml_backend_dayan_supports_quantized_glu(cgraph->nodes[cast_index]) &&
                ggml_backend_dayan_mxfp4_inputs_live(cgraph->nodes[cast_index], node, pair) &&
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

    ggml_backend_dayan_fuse_ssm_conv(cgraph, plan);
    return true;
}
