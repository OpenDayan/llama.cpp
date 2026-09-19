#pragma once

#include "ggml-dayan-memory.h"
#include "ggml-dayan-ffn.h"
#include "ggml-dayan-mxfp4.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static void run_router_f32(const DayanMatmulConfig * config, const float * activation, const float * weight, float * output) {
    constexpr uint32_t rows = 32, columns = 64, inner_tile = 256;
    constexpr uint32_t input_stride = 1152, output_stride = 384;
    const uint64_t base = (uintptr_t) dayan_uram_base();
    const uint64_t right = base + rows * input_stride;
    const uint64_t result = right + columns * input_stride;
    DAYAN_PARAMS_IF(!config || !activation || !weight || !output || !base ||
                    !config->m || !config->n || !config->k || result + rows * output_stride - base > dayan_uram_size()) {
        DAYAN_PARAMS_ERROR();
    }
    const uint64_t local_mask = Constants::kUramWindowStride - 1;
    const uint64_t nt = (config->n + columns - 1) / columns;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(input_stride);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(input_stride);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(output_stride);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_TF32, BaseNPUInstBuilder::DT_MATIN_TF32));
    long checksum = 0;
    for (uint64_t tile = dayan_task_id(DAYAN_TASK_X); tile < ((config->m + rows - 1) / rows) * nt; tile += dayan_task_dim(DAYAN_TASK_X)) {
        const uint64_t row = tile / nt * rows, column = tile % nt * columns;
        const uint32_t m = config->m - row < rows ? config->m - row : rows;
        const uint32_t n = config->n - column < columns ? config->n - column : columns;
        __tx_mat_tcsr_write<MAT_M>(m - 1);
        __tx_mat_tcsr_write<MAT_N>(n - 1);
        for (uint64_t inner = 0; inner < config->k; inner += inner_tile) {
            const uint32_t k = config->k - inner < inner_tile ? config->k - inner : inner_tile;
            set_lsu_matrix<0>(checksum, m, input_stride, config->a_stride, input_stride);
            __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
            load_matrix<0>(k * sizeof(float), m, input_stride, m, base,
                           (uintptr_t) activation + row * config->a_stride + inner * sizeof(float));
            set_lsu_matrix<0>(checksum, n, input_stride, config->bt_stride, input_stride);
            load_matrix<0>(k * sizeof(float), n, input_stride, n, right,
                           (uintptr_t) weight + column * config->bt_stride + inner * sizeof(float));
            dayan_wait_lsu();
            __tx_mat_tcsr_write<MAT_K>(k - 1);
            if (inner == 0) { __tx_mat_mm<0>(result & local_mask, base & local_mask, right & local_mask); }
            else { __tx_mat_mma<0>(result & local_mask, base & local_mask, right & local_mask); }
            dayan_wait_mat();
        }
        set_lsu_matrix<0>(checksum, m, output_stride, output_stride, config->output_stride);
        __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
        store_matrix<0>(n * sizeof(float), m, output_stride, m, result,
                        (uintptr_t) output + row * config->output_stride + column * sizeof(float));
        dayan_wait_lsu();
    }
}

struct SortUramLayout {
    enum : uint64_t {
        kLanes = (uint64_t) kDayanSortMaxWidth * sizeof(uint32_t),
        kCounter = kLanes + 256,
        kTotal = kCounter + 256,
    };
};

static_assert(SortUramLayout::kTotal <= DAYAN_SORT_TASK_URAM_BYTES, "Sort URAM layout is too large");

static inline bool moe_allocation_fits(uint64_t element_size, uint64_t ne0, uint64_t ne1, uint64_t ne2) {
    uint64_t       size          = element_size;
    const uint64_t dimensions[3] = { ne0, ne1, ne2 };
    for (int index = 0; index < 3; ++index) {
        if (dimensions[index] == 0 || dimensions[index] > UINT64_MAX / size) {
            return false;
        }
        size *= dimensions[index];
    }
    return true;
}

static inline int32_t moe_read_expert(const int32_t * ids, uint64_t index) {
    const uint64_t scratch = (uint64_t) (uintptr_t) dayan_uram_base();
    DAYAN_PARAMS_IF(scratch == 0 || dayan_uram_size() < sizeof(int32_t)) {
        DAYAN_PARAMS_ERROR();
    }
    return (int32_t) memory_read_index(ids, (uint32_t) index, kDayanDataTypeI32, scratch);
}

static inline uint64_t moe_dispatch_base() {
    return (uint64_t) (uintptr_t) shared + MATMUL_DAYAN_4C_ATOMIC_BYTES;
}

static inline volatile uint32_t * moe_dispatch_counts() {
    return (volatile uint32_t *) (uintptr_t) moe_dispatch_base();
}

static inline uint32_t * moe_dispatch_positions() {
    return (uint32_t *) (uintptr_t) (moe_dispatch_base() + (uint64_t) kDayanSortMaxWidth * sizeof(uint32_t));
}

static inline uint32_t * moe_dispatch_offsets(uint32_t total) {
    return (uint32_t *) (uintptr_t) (moe_dispatch_base() + (uint64_t) kDayanSortMaxWidth * sizeof(uint32_t) +
                                      (uint64_t) total * sizeof(uint32_t));
}

static inline bool moe_dispatch_fits(uint32_t experts, uint32_t total) {
    const uint64_t end = (uint64_t) MATMUL_DAYAN_4C_ATOMIC_BYTES +
                         (uint64_t) kDayanSortMaxWidth * sizeof(uint32_t) +
                         (uint64_t) (total + experts + 1U) * sizeof(uint32_t);
    return shared != nullptr && end <= DAYAN_SORT_ATOMIC_BYTES;
}

static inline void moe_copy_mxfp4_row(void * destination, const DayanMxfp4Layout & destination_layout,
                                      uint32_t destination_row,
                                      void * destination_scales, const DayanMxfp4Layout & source_layout,
                                      uint32_t source_row, const void * source, const void * source_scales,
                                      uint64_t source_scale_stride) {
    memory_copy_device((char *) destination + (uint64_t) destination_row * destination_layout.row_stride,
                       (const char *) source + (uint64_t) source_row * source_layout.row_stride,
                       destination_layout.row_stride);
    const uint32_t planes = (uint32_t) ((source_layout.columns + 127U) / 128U);
    for (uint32_t plane = 0; plane < planes; ++plane) {
        // TODO: replace these short D2D copies with one address-table Scatter when the firmware wrapper is exposed.
        memory_copy_device((char *) destination_scales + (uint64_t) plane * destination_layout.scale_stride +
                               (uint64_t) destination_row * 4,
                           (const char *) source_scales + (uint64_t) plane * source_scale_stride +
                               (uint64_t) source_row * 4,
                           4);
    }
}

static inline void moe_copy_output_row(void * destination, uint64_t destination_stride, void * destination_scales,
                                       uint64_t destination_scale_stride, const DayanMxfp4Layout & destination_layout,
                                       uint32_t destination_row, const void * source, uint64_t source_stride,
                                       const void * source_scales, const DayanMxfp4Layout & source_layout,
                                       uint32_t source_row, uint32_t output_type) {
    if (output_type == kDayanDataTypeMXFP4) {
        memory_copy_device((char *) destination + (uint64_t) destination_row * destination_stride,
                           (const char *) source + (uint64_t) source_row * source_layout.row_stride,
                           source_layout.row_stride);
        const uint32_t planes = (uint32_t) ((source_layout.columns + 127U) / 128U);
        for (uint32_t plane = 0; plane < planes; ++plane) {
            // TODO: use a table-based D2D Scatter for payload and scale rows.
            memory_copy_device((char *) destination_scales + (uint64_t) plane * destination_scale_stride +
                                   (uint64_t) destination_row * 4,
                               (const char *) source_scales + (uint64_t) plane * source_layout.scale_stride +
                                   (uint64_t) source_row * 4,
                               4);
        }
    } else {
        memory_copy_device((char *) destination + (uint64_t) destination_row * destination_stride,
                           (const char *) source + (uint64_t) source_row * source_stride,
                           source_stride);
    }
    (void) destination_layout;
}

static inline int run_router_indexer_f32(const DayanRouterIndexerConfig * config, const float * activation,
                                         const float * weight, float * logits, int32_t * selected) {
    if (!config || !activation || !weight || !logits || !selected || !config->matmul.m || !config->matmul.n ||
        !config->matmul.k || !config->experts_used || config->matmul.n > kDayanSortMaxWidth ||
        config->experts_used > config->matmul.n ||
        config->selected_stride < (uint64_t) config->experts_used * sizeof(int32_t)) {
        return -22;
    }
    run_router_f32(&config->matmul, activation, weight, logits);
    dayan_cluster_barrier_t * barrier =
        (dayan_cluster_barrier_t *) ((char *) shared + DAYAN_SORT_ATOMIC_BYTES);
    int status = dayan_cluster_barrier(4, barrier);
    if (status < 0) { return status; }

    const bool grouped = config->grouped != 0;
    const uint32_t total = (uint32_t) (config->matmul.m * config->experts_used);
    if (grouped && !moe_dispatch_fits((uint32_t) config->matmul.n, total)) {
        return -22;
    }
    volatile uint32_t * counts = grouped ? moe_dispatch_counts() : nullptr;
    uint32_t * positions = grouped ? moe_dispatch_positions() : nullptr;
    uint32_t * offsets = grouped ? moe_dispatch_offsets(total) : nullptr;
    if (grouped && dayan_native_task_id() == 0) {
        for (uint32_t expert = 0; expert < config->matmul.n; ++expert) {
            counts[expert] = 0;
        }
    }
    status = dayan_cluster_barrier(4, barrier);
    if (status < 0) { return status; }

    const uint32_t task = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    // TODO: replace the scalar per-token top-k scan with a FlexVec/radix selection pass.
    for (uint32_t token = task; token < config->matmul.m; token += tasks) {
        int32_t * row = (int32_t *) ((char *) selected + (uint64_t) token * config->selected_stride);
        for (uint32_t rank = 0; rank < config->experts_used; ++rank) {
            float best = -3.402823466e+38F;
            int32_t best_expert = -1;
            for (uint32_t expert = 0; expert < config->matmul.n; ++expert) {
                bool already = false;
                for (uint32_t previous = 0; previous < rank; ++previous) {
                    already = already || row[previous] == (int32_t) expert;
                }
                const float value = ((const float *) ((const char *) logits +
                                                       (uint64_t) token * config->matmul.output_stride))[expert];
                if (!already && (best_expert < 0 || value > best)) {
                    best = value;
                    best_expert = (int32_t) expert;
                }
            }
            if (best_expert < 0) { return -22; }
            row[rank] = best_expert;
            if (grouped) {
                const uint32_t assignment = token * config->experts_used + rank;
                positions[assignment] = __atomic_fetch_add((uint32_t *) &counts[best_expert], 1U, __ATOMIC_RELAXED);
            }
        }
    }
    status = dayan_cluster_barrier(4, barrier);
    if (status < 0) { return status; }
    if (!grouped) {
        return dayan_cluster_barrier(4, barrier);
    }
    if (dayan_native_task_id() == 0) {
        // TODO: compute this expert-count prefix with FlexVec Shuffle instead of the scalar carry loop.
        uint32_t running = 0;
        for (uint32_t expert = 0; expert < config->matmul.n; ++expert) {
            offsets[expert] = running;
            running += counts[expert];
        }
        offsets[config->matmul.n] = running;
    }
    return dayan_cluster_barrier(4, barrier);
}

static int run_mul_mat_id_mxfp4_grouped(const DayanMxfp4MoeConfig * config, const void * a, const void * as,
                                         const void * b, const void * bs, const void * up, const void * ups,
                                         const int32_t * ids, void * output, void * scales, void * workspace) {
    if (!config) {
        return -22;
    }
    const auto & r = config->routing;
    const auto & mm = config->matmul;
    const uint32_t total = r.tokens * r.experts_used;
    const DayanMxfp4Layout input_layout = DayanMxfp4Layout::make(total, mm.k);
    const DayanMxfp4Layout output_layout = DayanMxfp4Layout::make(total, mm.n);
    const uint64_t output_bytes = mm.output_type == kDayanDataTypeMXFP4 ? output_layout.bytes :
                                                                            (uint64_t) total * mm.n * sizeof(float);
    const uint64_t output_offset = (input_layout.bytes + 63U) & ~63ULL;
    const uint64_t required = output_offset + ((output_bytes + 63U) & ~63ULL);
    if (!config || !workspace || !shared || !a || !as || !b || !bs || !up || !ups || !ids || !output ||
        !r.tokens || !r.experts_used || !r.experts || !r.activation_slots || !config->workspace_bytes ||
        !moe_dispatch_fits(r.experts, total) || required > config->workspace_bytes || r.experts_used % r.activation_slots ||
        mm.m != 1 || mm.weight_type != kDayanDataTypeMXFP4 ||
        (mm.output_type == kDayanDataTypeMXFP4 && !scales)) {
        return -22;
    }
    uint32_t * positions = moe_dispatch_positions();
    uint32_t * offsets = moe_dispatch_offsets(total);
    if (offsets[r.experts] != total) { return -22; }
    const uint64_t grouped_a = (uintptr_t) workspace;
    const uint64_t grouped_as = grouped_a + input_layout.scale_offset;
    const uint64_t grouped_output = grouped_a + output_offset;
    const uint64_t grouped_output_scales = grouped_output + output_layout.scale_offset;
    const uint32_t task = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    const DayanMxfp4Layout source_layout = DayanMxfp4Layout::make(r.tokens * r.activation_slots, mm.k);
    for (uint32_t assignment = task; assignment < total; assignment += tasks) {
        // TODO: move address generation into a D2D Scatter address table; scalar code currently owns this mapping.
        const uint32_t token = assignment / r.experts_used;
        const uint32_t used = assignment % r.experts_used;
        const int32_t expert = moe_read_expert((const int32_t *) ((const char *) ids + (uint64_t) token * r.ids_row_stride), used);
        if (expert < 0 || (uint32_t) expert >= r.experts) { return -22; }
        const uint32_t group_row = offsets[expert] + positions[assignment];
        const uint32_t input_row = token * r.activation_slots + used % r.activation_slots;
        moe_copy_mxfp4_row((void *) grouped_a, input_layout, group_row, (void *) grouped_as,
                           source_layout, input_row, a, as, mm.a_scale_stride);
    }
    dayan_cluster_barrier_t * barrier =
        (dayan_cluster_barrier_t *) ((char *) shared + DAYAN_SORT_ATOMIC_BYTES);
    int status = dayan_cluster_barrier(4, barrier);
    if (status < 0) { return status; }

    ggml_dayan_mxfp4_detail::Phases phases = {};
    DayanMxfp4Config batch = mm;
    batch.a_stride = input_layout.row_stride;
    batch.a_scale_stride = input_layout.scale_stride;
    batch.output_stride = mm.output_type == kDayanDataTypeMXFP4 ? output_layout.row_stride : mm.n * sizeof(float);
    batch.output_scale_stride = mm.output_type == kDayanDataTypeMXFP4 ? output_layout.scale_stride : 0;
    for (uint32_t expert = 0; expert < r.experts; ++expert) {
        const uint32_t begin = offsets[expert];
        const uint32_t count = offsets[expert + 1] - begin;
        if (!count) { continue; }
        batch.m = count;
        status = ggml_dayan_mxfp4_detail::run(
            &batch, (const void *) (grouped_a + (uint64_t) begin * input_layout.row_stride),
            (const void *) (grouped_as + (uint64_t) begin * 4),
            (const char *) b + (uint64_t) expert * mm.n * mm.b_stride,
            (const char *) bs + (uint64_t) expert * mm.n * 4,
            (const char *) up + (uint64_t) expert * mm.n * mm.b_stride,
            (const char *) ups + (uint64_t) expert * mm.n * 4,
            (void *) (grouped_output + (uint64_t) begin * batch.output_stride),
            mm.output_type == kDayanDataTypeMXFP4 ? (void *) (grouped_output_scales + (uint64_t) begin * 4) : nullptr,
            &phases);
        if (status < 0) { return status; }
    }
    status = dayan_cluster_barrier(4, barrier);
    if (status < 0) { return status; }
    // TODO: use one table-based D2D Scatter for the inverse output mapping once the address-table wrapper is available.
    for (uint32_t assignment = task; assignment < total; assignment += tasks) {
        const uint32_t token = assignment / r.experts_used;
        const uint32_t used = assignment % r.experts_used;
        const int32_t expert = moe_read_expert((const int32_t *) ((const char *) ids + (uint64_t) token * r.ids_row_stride), used);
        if (expert < 0 || (uint32_t) expert >= r.experts) { return -22; }
        const uint32_t group_row = offsets[expert] + positions[assignment];
        moe_copy_output_row(output, mm.output_stride, scales, mm.output_scale_stride, output_layout, assignment,
                            (const void *) grouped_output, batch.output_stride,
                            mm.output_type == kDayanDataTypeMXFP4 ? (const void *) grouped_output_scales : nullptr,
                            output_layout, group_row, mm.output_type);
    }
    return dayan_cluster_barrier(4, barrier);
}

static int run_mul_mat_id_mxfp4(const DayanMxfp4MoeConfig * config, const void * a, const void * as,
                               const void * b, const void * bs, const void * up, const void * ups,
                               const int32_t * ids, void * output, void * scales, void * workspace) {
    DAYAN_PARAMS_IF(!config || !a || !as || !b || !bs || !up || !ups || !ids || !output || !scales) { return -22; }
    if (config->grouped) {
        return run_mul_mat_id_mxfp4_grouped(config, a, as, b, bs, up, ups, ids, output, scales, workspace);
    }
    const auto & r = config->routing;
    const auto & mm = config->matmul;
    DAYAN_PARAMS_IF(!r.k || r.k % 32 || !r.n || r.n % 16 || !r.experts || !r.experts_used || !r.tokens ||
        !r.activation_slots || r.experts_used % r.activation_slots || r.ids_row_stride < r.experts_used * sizeof(int32_t) ||
        mm.m != 1 || mm.n != r.n || mm.k != r.k || mm.weight_type != kDayanDataTypeMXFP4) { return -22; }
    ggml_dayan_mxfp4_detail::Phases phases;
    for (uint32_t token = 0; token < r.tokens; ++token) {
        const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * r.ids_row_stride);
        for (uint32_t used = 0; used < r.experts_used; ++used) {
            const int32_t expert = moe_read_expert(row_ids, used);
            if (expert < 0 || (uint32_t) expert >= r.experts) { return -22; }
            const uint64_t input_row = (uint64_t) token * r.activation_slots + used % r.activation_slots;
            const uint64_t weight_row = (uint64_t) expert * r.n;
            const uint64_t output_row = (uint64_t) token * r.experts_used + used;
            const int status = ggml_dayan_mxfp4_detail::run(&mm,
                (const char *) a + input_row * mm.a_stride, (const char *) as + input_row * 4,
                (const char *) b + weight_row * mm.b_stride, (const char *) bs + weight_row * 4,
                (const char *) up + weight_row * mm.b_stride, (const char *) ups + weight_row * 4,
                (char *) output + output_row * mm.output_stride, (char *) scales + output_row * 4, &phases);
            if (status < 0) { return status; }
        }
    }
    return 0;
}

template <typename Format>
static void run_mul_mat_id(const DayanMulMatIdConfig * config,
                           const uint16_t *            activation,
                           const uint16_t *            weight,
                           const int32_t *             ids,
                           float *                     output, MatmulPhases * phases = nullptr) {
    DAYAN_PARAMS_IF(config == nullptr || activation == nullptr || weight == nullptr || ids == nullptr || output == nullptr ||
        config->k == 0 || config->n == 0 || config->experts == 0 || config->experts > INT32_MAX ||
        config->experts_used == 0 || config->tokens == 0 || config->activation_slots == 0 ||
        config->ids_row_stride < (uint64_t) config->experts_used * sizeof(int32_t) ||
        config->ids_row_stride > UINT64_MAX / config->tokens ||
        config->experts_used % config->activation_slots != 0 ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->n, config->experts) ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->activation_slots, config->tokens) ||
        !moe_allocation_fits(sizeof(int32_t), config->experts_used, config->tokens, 1) ||
        !moe_allocation_fits(sizeof(float), config->n, config->experts_used, config->tokens)) {
        DAYAN_PARAMS_ERROR();
    }

    const DayanMatmulConfig matmul = {
        1,
        config->n,
        config->k,
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->n * sizeof(float),
    };
    const uint64_t weight_elements = (uint64_t) config->n * config->k;
    MatmulPhases local_phases;
    MatmulPhases & resource_phases = phases ? *phases : local_phases;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * config->ids_row_stride);
            const int32_t expert = moe_read_expert(row_ids, used);
            if (expert < 0 || expert >= (int32_t) config->experts) {
                DAYAN_PARAMS_ERROR();
            }
            const uint32_t activation_slot = used % config->activation_slots;
            const uint64_t activation_offset =
                ((uint64_t) token * config->activation_slots + activation_slot) * config->k;
            const uint64_t weight_offset = (uint64_t) expert * weight_elements;
            const uint64_t output_offset = ((uint64_t) token * config->experts_used + used) * config->n;
            matmul_check(run_matmul<Format, 1, 1, sizeof(float), false, epilogue_identity, nullptr, 0, true>(
                &matmul, activation + activation_offset, weight + weight_offset, nullptr, output + output_offset, nullptr, &resource_phases));
        }
    }
}

template <typename Format, bool GeGLU>
static void run_mul_mat_id_glu(const DayanMulMatIdConfig * config,
                               const uint16_t *            activation,
                               const uint16_t *            gate_weight,
                               const uint16_t *            up_weight,
                               const int32_t *             ids,
                               uint16_t *                  output) {
    DAYAN_PARAMS_IF(config == nullptr || activation == nullptr || gate_weight == nullptr || up_weight == nullptr ||
        ids == nullptr || output == nullptr || config->k == 0 || config->n == 0 ||
        config->experts == 0 || config->experts > INT32_MAX || config->experts_used == 0 || config->tokens == 0 ||
        config->ids_row_stride < (uint64_t) config->experts_used * sizeof(int32_t) ||
        config->ids_row_stride > UINT64_MAX / config->tokens ||
        config->activation_slots == 0 || config->experts_used % config->activation_slots != 0 ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->n, config->experts) ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->activation_slots, config->tokens) ||
        !moe_allocation_fits(sizeof(int32_t), config->experts_used, config->tokens, 1) ||
        !moe_allocation_fits(sizeof(uint16_t), config->n, config->experts_used, config->tokens)) {
        DAYAN_PARAMS_ERROR();
    }

    const DayanMatmulConfig matmul = {
        1,
        config->n,
        config->k,
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->n * sizeof(uint16_t),
    };
    const uint64_t weight_elements = (uint64_t) config->n * config->k;
    MatmulPhases resource_phases;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * config->ids_row_stride);
            const int32_t expert = moe_read_expert(row_ids, used);
            if (expert < 0 || expert >= (int32_t) config->experts) {
                DAYAN_PARAMS_ERROR();
            }
            const uint32_t activation_slot = used % config->activation_slots;
            const uint64_t activation_offset =
                ((uint64_t) token * config->activation_slots + activation_slot) * config->k;
            const uint64_t weight_offset = (uint64_t) expert * weight_elements;
            const uint64_t output_offset = ((uint64_t) token * config->experts_used + used) * config->n;
            matmul_check(ggml_dayan_ffn_detail::run<Format, 1, GeGLU, true>(
                &matmul, activation + activation_offset, gate_weight + weight_offset, up_weight + weight_offset,
                output + output_offset, &resource_phases));
        }
    }
}

static inline void run_add_id(const DayanAddIdConfig * config,
                              const float *            input,
                              const float *            bias,
                              const int32_t *          ids,
                              float *                  output) {
    if (config == nullptr || input == nullptr || bias == nullptr || ids == nullptr || output == nullptr ||
        config->width == 0 || config->experts == 0 || config->experts > INT32_MAX || config->experts_used == 0 ||
        config->tokens == 0 || config->ids_row_stride < (uint64_t) config->experts_used * sizeof(int32_t) ||
        !moe_allocation_fits(sizeof(float), config->width, config->experts_used, config->tokens) ||
        !moe_allocation_fits(sizeof(float), config->width, config->experts, 1) ||
        !moe_allocation_fits(sizeof(int32_t), config->experts_used, config->tokens, 1)) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate((uint64_t) config->width * config->experts_used * config->tokens, input, output);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t bias_uram  = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    const uint64_t rows       = (uint64_t) config->tokens * config->experts_used;
    for (uint64_t row = task; row < rows; row += tasks) {
        const uint32_t used   = (uint32_t) (row % config->experts_used);
        const uint32_t token  = (uint32_t) (row / config->experts_used);
        const int32_t * token_ids =
            (const int32_t *) ((const char *) ids + (uint64_t) token * config->ids_row_stride);
        const int32_t expert = moe_read_expert(token_ids, used);
        if (expert < 0 || expert >= (int32_t) config->experts) {
            DAYAN_PARAMS_ERROR();
        }
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input + row * config->width + offset, elements);
            normalization_load(bias_uram, bias + (uint64_t) expert * config->width + offset, elements);
            ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
            constexpr long instructions = 4;
            ggml_dayan_vec_detail::begin_vec_block(instructions);
            __tx_vec_lsu_u2v<0, 0>((long) input_uram);
            __tx_vec_lsu_u2v<1, 1>((long) bias_uram);
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 3>((long) out_uram);
            ggml_dayan_vec_detail::start_vec_block(instructions);
            dayan_wait_vec();
            normalization_store_f32(output + row * config->width + offset, out_uram, elements);
            dayan_wait_lsu();
        }
    }
}

template <int Shift> static inline void sort_scan_step() {
    __tx_vec_lsu_g2s<0>((long) (~uint64_t{0} << Shift));
    __tx_vec_shuffle_ror<0, 0>(64 - Shift);
    __tx_vec_vv_int32_add<0, 1, 0>();
    __tx_vec_vv_mux2<0, 2, 0, 0>();
}

template <bool TopK> static void run_sort(const DayanSortConfig * config, const float * input, int32_t * output) {
    using namespace ggml_dayan_vec_detail;
    memory_validate(config);
    DAYAN_PARAMS_IF(!input || !output || !config->width || !config->rows || !config->output_width ||
                    config->width > kDayanSortMaxWidth || config->output_width > config->width || config->order > 1 ||
                    config->output_stride < config->output_width || (TopK && config->order != kDayanSortOrderDesc) ||
                    !dayan_uram_base() || dayan_uram_size() < SortUramLayout::kTotal || dayan_native_task_count() > 16) {
        DAYAN_PARAMS_ERROR();
    }
    const uint32_t task = dayan_native_task_id(), tasks = dayan_native_task_count();
    const uint64_t uram = (uintptr_t) dayan_uram_base();
    const uint64_t lanes = uram + SortUramLayout::kLanes, counter = uram + SortUramLayout::kCounter;
    constexpr uint64_t plane = kDayanSortMaxWidth * sizeof(uint32_t);
    const uint64_t scratch = (uintptr_t) shared + task * 4 * plane;
    for (uint32_t i = 0; i < 64; ++i) { ((uint32_t *) lanes)[i] = i; }
    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    const uint32_t padded = (config->width + 63) & ~63U;
    for (uint32_t row = task; row < config->rows; row += tasks) {
        memory_load_bytes(uram, input + (uint64_t) row * config->width, config->width * sizeof(float));
        for (uint32_t first = 0; first < padded; first += 64) {
            const uint32_t valid = config->width - first < 64 ? config->width - first : 64;
            begin_vec_block(19);
            __tx_vec_lsu_u2v<0, 0>(uram + first * 4);
            __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
            __tx_vec_vv_fp32_cmp<0, 16, 0, 0b010>();
            __tx_vec_vv_mux2<0, 0, 16, 0>();
            __tx_vec_lsu_g2v_broadcast32<1, 1>(31);
            __tx_vec_vv_int32_sra<0, 0, 16>();
            __tx_vec_lsu_g2v_broadcast32<1, 1>(0x80000000U);
            __tx_vec_vv_int32_bitwise<0, 0, 16, 0b1110>();
            __tx_vec_vv_int32_bitwise<0, 2, 0, 0b0110>();
            __tx_vec_lsu_g2v_broadcast32<1, 1>(config->order == kDayanSortOrderDesc ? UINT32_MAX : 0);
            __tx_vec_vv_int32_bitwise<0, 0, 16, 0b0110>();
            __tx_vec_lsu_g2v_broadcast32<1, 1>(UINT32_MAX);
            __tx_vec_lsu_g2s<0>(valid == 64 ? UINT64_MAX : (uint64_t{1} << valid) - 1);
            __tx_vec_vv_mux2<0, 16, 0, 0>();
            __tx_vec_lsu_v2sm<0, 0>(scratch + first * 4);
            __tx_vec_lsu_u2v<0, 0>(lanes);
            __tx_vec_lsu_g2v_broadcast32<1, 1>(first);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_lsu_v2sm<0, 0>(scratch + plane + first * 4);
            start_vec_block(19);
            dayan_wait_vec();
        }
        uint64_t source = scratch, destination = scratch + 2 * plane;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            for (uint32_t first = 0; first < padded; first += 64) {
                begin_vec_block(36);
                __tx_vec_lsu_sm2v<0, 0>(source + first * 4);
                __tx_vec_lsu_g2v_broadcast32<1, 1>(uint32_t{1} << bit);
                __tx_vec_vv_int32_bitwise<0, 0, 16, 0b1000>();
                __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
                __tx_vec_vv_int32_cmp<0, 16, 0, 0b010>();
                __tx_vec_lsu_g2v_broadcast32<1, 1>(1);
                __tx_vec_vv_mux2<0, 17, 16, 0>();
                sort_scan_step<1>();
                sort_scan_step<2>();
                sort_scan_step<4>();
                sort_scan_step<8>();
                sort_scan_step<16>();
                sort_scan_step<32>();
                if (first == 0) { __tx_vec_lsu_g2v_broadcast32<1, 1>(0); }
                else { __tx_vec_lsu_u2v<1, 1>(counter); }
                __tx_vec_vv_int32_add<0, 0, 16>();
                __tx_vec_lsu_v2u<0, 0>(uram + first * 4);
                __tx_vec_shuffle_broadcast<0, 0>(63);
                __tx_vec_lsu_v2u<0, 0>(counter);
                start_vec_block(36);
                dayan_wait_vec();
            }
            for (uint32_t first = 0; first < padded; first += 64) {
                begin_vec_block(22);
                __tx_vec_lsu_u2v<0, 0>(uram + first * 4);
                __tx_vec_lsu_g2v_broadcast32<1, 1>(1);
                __tx_vec_vv_int32_sub<0, 0, 16>();
                __tx_vec_lsu_sm2v<1, 0>(source + first * 4);
                __tx_vec_lsu_g2v_broadcast32<1, 1>(uint32_t{1} << bit);
                __tx_vec_vv_int32_bitwise<1, 17, 16, 0b1000>();
                __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
                __tx_vec_vv_int32_cmp<17, 16, 0, 0b010>();
                __tx_vec_lsu_u2v<1, 0>(lanes);
                __tx_vec_lsu_g2v_broadcast32<1, 1>(first);
                __tx_vec_vv_int32_add<1, 17, 16>();
                __tx_vec_vv_int32_sub<0, 16, 1>();
                __tx_vec_lsu_u2v<1, 1>(counter);
                __tx_vec_vv_int32_add<0, 0, 16>();
                __tx_vec_vv_mux2<0, 0, 2, 0>();
                __tx_vec_lsu_g2v_broadcast32<1, 1>(2);
                __tx_vec_vv_int32_sll<0, 0, 16>();
                __tx_vec_lsu_sm2v<1, 0>(source + first * 4);
                __tx_vec_lsu_g2s<0>(UINT64_MAX);
                __tx_vec_v2sm_sparse<16, 0, 0>(destination);
                __tx_vec_lsu_sm2v<1, 0>(source + plane + first * 4);
                __tx_vec_v2sm_sparse<16, 0, 0>(destination + plane);
                start_vec_block(22);
                dayan_wait_vec();
            }
            const uint64_t previous = source;
            source = destination;
            destination = previous;
        }
        configure_vec_grid((config->output_width + 63) / 64);
        configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, 0);
        configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, 0);
        begin_vec_block(2);
        __tx_vec_lsu_sm2v<0, 0>(source + plane);
        __tx_vec_lsu_v2u<0, 3>(uram);
        start_vec_block(2);
        dayan_wait_vec();
        configure_vec_grid(1);
        memory_store_bytes(output + (uint64_t) row * config->output_stride, uram, config->output_width * sizeof(int32_t));
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif
