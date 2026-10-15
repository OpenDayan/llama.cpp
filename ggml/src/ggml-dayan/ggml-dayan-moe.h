#pragma once

#include "ggml-dayan-memory.h"
#include "ggml-dayan-ffn.h"
#include "ggml-dayan-mxfp4.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static void run_router_f32(const DayanMatmulConfig * config, const float * activation, const float * weight, float * output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {(uint64_t) config->m * config->n * config->k, ((config->m + 31) / 32) * ((config->n + 63) / 64), 256U * 1024U, 1, 1, 0}, [&](const NativeContext & worker) {
            run_router_f32(config, activation, weight, output, worker);
        })) { return; }

    constexpr uint32_t rows = 32, columns = 64, inner_tile = 256;
    constexpr uint32_t input_stride = 1152, output_stride = 384;
    const uint64_t base = native_uram(context);
    const uint64_t right = base + rows * input_stride;
    const uint64_t result = right + columns * input_stride;
    DAYAN_PARAMS_IF(!config || !activation || !weight || !output || !base ||
                    !config->m || !config->n || !config->k || result + rows * output_stride - base > native_uram_size(context)) {
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
    for (uint64_t tile = dayan_native_task_id(context); tile < ((config->m + rows - 1) / rows) * nt; tile += dayan_native_task_count(context)) {
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

template <unsigned Distance = 1> static inline void moe_prefix_sum() {
    __tx_vec_lsu_g2s<3>(UINT64_MAX << Distance);
    __tx_vec_shuffle_ror<1, 0>(64 - Distance);
    __tx_vec_vv_int32_add<0, 0, 16>();
    __tx_vec_vv_mux2<0, 1, 0, 3>();
    if constexpr (Distance < 32) { moe_prefix_sum<Distance * 2>(); }
}

static inline void moe_lane_ids(uint64_t output) {
    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    begin_vec_block(28);
    __tx_vec_lsu_g2v_broadcast32<0, 0>(1);
    moe_prefix_sum();
    __tx_vec_lsu_g2v_broadcast32<1, 0>(1);
    __tx_vec_vv_int32_sub<0, 0, 16>();
    __tx_vec_lsu_v2u<0, 0>(output);
    start_vec_block(28);
    dayan_wait_vec();
}

static constexpr uint64_t sort_lane_mask(unsigned length, unsigned distance) {
    uint64_t mask = 0;
    for (unsigned lane = 0; lane < 64; ++lane) {
        if (((lane & length) == 0) == ((lane & distance) == 0)) { mask |= uint64_t{1} << lane; }
    }
    return mask;
}

static inline void sort_compare() {
    // Q0 holds peer/own keys; Q1 holds peer/own indices. Break ties by index.
    __tx_vec_vv_int32_cmp<1, 0, 0, 0b100>();
    __tx_vec_vv_int32_cmp<1, 0, 1, 0b010>();
    __tx_vec_vv_int32_cmp<17, 16, 2, 0b100>();
    __tx_vec_s_bitwise<1, 1, 2, 0b1000>();
    __tx_vec_s_bitwise<0, 0, 1, 0b1110>();
}

template <unsigned Length, unsigned Distance> static inline void sort_lane_step(bool descending) {
    constexpr uint64_t mask = sort_lane_mask(Length, Distance);
    __tx_vec_shuffle_xor<0, 0>(Distance);
    __tx_vec_shuffle_xor<1, 16>(Distance);
    sort_compare();
    __tx_vec_lsu_g2s<3>(descending ? ~mask : mask);
    __tx_vec_s_bitwise<0, 0, 3, 0b0110>();
    __tx_vec_vv_mux2<0, 1, 0, 0>();
    __tx_vec_vv_mux2<1, 17, 16, 0>();
}

template <unsigned Length = 2, unsigned Distance = 1> static inline void sort_network(bool descending) {
    sort_lane_step<Length, Distance>(descending);
    if constexpr (Distance > 1) { sort_network<Length, Distance / 2>(descending); }
    else if constexpr (Length < 64) { sort_network<Length * 2, Length>(descending); }
}

template <unsigned Distance = 32> static inline void sort_merge_lanes(bool descending) {
    sort_lane_step<64, Distance>(descending);
    if constexpr (Distance > 1) { sort_merge_lanes<Distance / 2>(descending); }
}

template <bool Load> static void sort_transfer(uint64_t uram, uint64_t dram, uint32_t rows, uint32_t bytes,
                                              uint64_t uram_stride, uint64_t dram_stride) {
    const uint32_t whole = bytes & ~63U;
    if (whole) {
        configure_memory_lsu_3d<1>(whole / 64, 64, 64, rows, Load ? dram_stride : uram_stride,
                                  Load ? uram_stride : dram_stride, 1, 0, 0, 64, 64);
        if constexpr (Load) { __tx_lsu_dma_d2u<1>(uram, dram); }
        else { __tx_lsu_dma_u2d<1>(dram, uram); }
    }
    if (bytes != whole) {
        configure_memory_lsu<1>(rows, Load ? dram_stride : uram_stride, Load ? uram_stride : dram_stride,
                                bytes - whole, bytes - whole);
        if constexpr (Load) { __tx_lsu_dma_d2u<1>(uram + whole, dram + whole); }
        else { __tx_lsu_dma_u2d<1>(dram + whole, uram + whole); }
    }
    dayan_wait_lsu();
}

static void sort_chunks(uint64_t keys, uint64_t ids, uint64_t out_keys, uint64_t out_ids, uint64_t lanes,
                        uint32_t width, uint32_t rows, uint32_t input_stride, uint32_t output_stride,
                        uint32_t keep, uint32_t first = 0, uint32_t step = 1, bool descending = false, uint32_t span = 0) {
    const uint32_t chunks = ((span ? span : width) + 63) / 64;
    configure_vec_grid((chunks - first + step - 1) / step, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, step * 256, input_stride);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, step * keep * 4, output_stride);
    configure_vec_iterator<VEC_S0_BASE>(0, 0, step * 64, 0);
    __tx_vec_tcsr_write<VEC_S0_BASE>(first * 64);
    constexpr long instructions = 245;
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(keys + first * 256);
    __tx_vec_lsu_u2v<1, 0>(ids + first * 256);
    __tx_vec_lsu_u2v<0, 2>(lanes);
    __tx_vec_s_iter_gets<4, 0, 0, 0>();
    __tx_vec_lsu_s2v_broadcast32<0, 2, 4>();
    __tx_vec_vv_int32_add<0, 1, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 2>(width);
    __tx_vec_vv_int32_cmp<1, 0, 4, 0b100>();
    __tx_vec_lsu_g2v_broadcast32<0, 2>(INT32_MAX);
    __tx_vec_vv_mux2<0, 0, 5, 4>();
    __tx_vec_vv_mux2<1, 1, 16, 4>();
    sort_network(descending);
    __tx_vec_lsu_g2s<4>(keep == 64 ? UINT64_MAX : (uint64_t{1} << keep) - 1);
    __tx_vec_lsu_v2u_mask<0, 3, 4>(out_keys + first * keep * 4);
    __tx_vec_lsu_v2u_mask<16, 3, 4>(out_ids + first * keep * 4);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static void sort_full(uint64_t keys, uint64_t ids, uint64_t lanes, uint32_t width, uint32_t padded,
                      uint32_t rows, uint32_t stride) {
    sort_chunks(keys, ids, keys, ids, lanes, width, rows, stride, stride, 64, 0, 2, false, padded);
    if (padded > 64) { sort_chunks(keys, ids, keys, ids, lanes, width, rows, stride, stride, 64, 1, 2, true, padded); }
    for (uint32_t length = 128; length <= padded; length *= 2) {
        for (uint32_t distance = length / 2; distance >= 64; distance /= 2) {
            configure_vec_grid(distance / 64, rows);
            configure_vec_iterator<kVecV0Base>(0, 0, 256, stride);
            for (uint32_t first = 0; first < padded; first += 2 * distance) {
                const uint64_t offset = first * 4, peer = offset + distance * 4;
                constexpr long instructions = 20;
                begin_vec_block(instructions);
                __tx_vec_lsu_u2v<0, 0>(keys + offset);
                __tx_vec_lsu_u2v<0, 0>(keys + peer);
                __tx_vec_lsu_u2v<1, 0>(ids + offset);
                __tx_vec_lsu_u2v<1, 0>(ids + peer);
                sort_compare();
                __tx_vec_lsu_g2s<3>((first & length) ? 0 : UINT64_MAX);
                __tx_vec_s_bitwise<0, 0, 3, 0b0110>();
                __tx_vec_vv_mux2<0, 1, 0, 0>();
                __tx_vec_vv_mux2<1, 17, 16, 0>();
                __tx_vec_s_bitwise<0, 0, 0, 0b0011>();
                __tx_vec_vv_mux2<0, 2, 1, 0>();
                __tx_vec_vv_mux2<1, 18, 17, 0>();
                __tx_vec_lsu_v2u<1, 0>(keys + offset);
                __tx_vec_lsu_v2u<0, 0>(keys + peer);
                __tx_vec_lsu_v2u<17, 0>(ids + offset);
                __tx_vec_lsu_v2u<16, 0>(ids + peer);
                start_vec_block(instructions);
            }
            dayan_wait_vec();
        }
        configure_vec_grid(length / 64, rows);
        for (uint32_t first = 0; first < padded; first += length) {
            constexpr long instructions = 70;
            begin_vec_block(instructions);
            __tx_vec_lsu_u2v<0, 0>(keys + first * 4);
            __tx_vec_lsu_u2v<1, 0>(ids + first * 4);
            sort_merge_lanes((first & length) != 0);
            __tx_vec_lsu_v2u<0, 0>(keys + first * 4);
            __tx_vec_lsu_v2u<16, 0>(ids + first * 4);
            start_vec_block(instructions);
        }
        dayan_wait_vec();
    }
}

template <bool TopK> static void run_sort(const DayanSortConfig * config, const float * input, int32_t * output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {(uint64_t) config->width * config->rows, config->rows, 256U * 1024U, 2, 0, 1}, [&](const NativeContext & worker) {
            run_sort<TopK>(config, input, output, worker);
        })) { return; }
    memory_validate(config, context);
    DAYAN_PARAMS_IF(!input || !output || !config->width || !config->rows || !config->output_width ||
                    config->width > kDayanSortMaxWidth || config->output_width > config->width || config->order > 1 ||
                    config->output_stride < config->output_width || (TopK && config->order != kDayanSortOrderDesc)) {
        DAYAN_PARAMS_ERROR();
    }
    const uint32_t task = dayan_native_task_id(context), tasks = dayan_native_task_count(context);
    const uint64_t per_task = ((uint64_t) config->rows + tasks - 1) / tasks;
    const uint32_t begin = task * per_task < config->rows ? task * per_task : config->rows;
    const uint32_t end = begin + per_task < config->rows ? begin + per_task : config->rows;
    if (begin == end) { return; }
    // Keep each candidate group aligned to a 32-byte URAM channel.
    uint32_t padded = 64, keep = 8;
    while (padded < config->width) { padded *= 2; }
    while (keep < config->output_width) { keep *= 2; }
    const bool partial = keep <= 32 && config->output_width < config->width;
    const uint32_t stride = padded * 4;
    const uint32_t candidate_stride = partial ? ((padded / 64 * keep + 63) & ~63U) * 4 : 0;
    const uint32_t capacity = (native_uram_size(context) - 256) / (2 * (stride + candidate_stride));
    const uint64_t lanes = native_uram(context), keys = lanes + 256, ids = keys + capacity * stride;
    const uint64_t candidates = ids + capacity * stride, candidate_ids = candidates + capacity * candidate_stride;
    moe_lane_ids(lanes);
    for (uint32_t row = begin; row < end; row += capacity) {
        const uint32_t rows = end - row < capacity ? end - row : capacity;
        sort_transfer<true>(keys, (uintptr_t) (input + (uint64_t) row * config->width), rows, config->width * 4, stride, config->width * 4);
        configure_vec_grid(padded / 64, rows);
        configure_vec_iterator<kVecV0Base>(0, 0, 256, stride);
        configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
        configure_vec_iterator<VEC_S0_BASE>(0, 0, 64, 0);
        begin_vec_block(16);
        __tx_vec_lsu_u2v<0, 0>(keys);
        __tx_vec_lsu_g2v_broadcast32<1, 2>(0);
        __tx_vec_vv_fp32_cmp<0, 16, 0, 0b010>();
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_lsu_g2v_broadcast32<1, 2>(0x7fffffffU);
        __tx_vec_v_int32_sra<0, 0, 31>();
        __tx_vec_vv_int32_bitwise<0, 0, 16, 0b1000>();
        __tx_vec_vv_int32_bitwise<0, 2, 0, 0b0110>();
        __tx_vec_lsu_g2v_broadcast32<1, 2>(config->order == kDayanSortOrderDesc ? UINT32_MAX : 0);
        __tx_vec_vv_int32_bitwise<0, 0, 16, 0b0110>();
        __tx_vec_lsu_v2u<0, 0>(keys);
        __tx_vec_lsu_u2v<0, 2>(lanes);
        __tx_vec_s_iter_gets<0, 0, 0, 0>();
        __tx_vec_lsu_s2v_broadcast32<1, 2, 0>();
        __tx_vec_vv_int32_add<1, 0, 16>();
        __tx_vec_lsu_v2u<16, 0>(ids);
        start_vec_block(16);
        dayan_wait_vec();
        uint64_t result = ids;
        uint32_t result_stride = stride;
        if (partial) {
            uint64_t source = keys, source_ids = ids, destination = candidates, destination_ids = candidate_ids;
            uint32_t source_stride = stride, destination_stride = candidate_stride, width = config->width;
            for (;;) {
                sort_chunks(source, source_ids, destination, destination_ids, lanes, width, rows, source_stride, destination_stride, keep);
                result = destination_ids;
                result_stride = destination_stride;
                if (width <= 64) { break; }
                width = ((width + 63) / 64) * keep;
                const uint64_t old_source = source, old_ids = source_ids;
                const uint32_t old_stride = source_stride;
                source = destination; source_ids = destination_ids; source_stride = destination_stride;
                destination = old_source; destination_ids = old_ids; destination_stride = old_stride;
            }
        } else {
            sort_full(keys, ids, lanes, config->width, padded, rows, stride);
        }
        sort_transfer<false>(result, (uintptr_t) (output + (uint64_t) row * config->output_stride), rows, config->output_width * 4,
                             result_stride, config->output_stride * 4);
    }
}

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

// SparseMem is local to VecTop. Mega owns the core's Proc slots during this stage.
static constexpr uint32_t kMoeExpertWindow = 1024;
static constexpr long kMoeHistogram = 0;
static constexpr long kMoeReorder = kMoeExpertWindow * sizeof(uint32_t);
static_assert(kMoeReorder + 256 <= 8192, "MoE dispatch exceeds local SparseMem");

template <unsigned Distance = 1> static inline void moe_prefix_max() {
    __tx_vec_lsu_g2s<1>(UINT64_MAX << Distance);
    __tx_vec_shuffle_ror<1, 0>(64 - Distance);
    __tx_vec_vv_int32_cmp<0, 16, 0, 0b100>();
    __tx_vec_s_bitwise<0, 0, 1, 0b1000>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    if constexpr (Distance < 32) { moe_prefix_max<Distance * 2>(); }
}

static inline int run_moe_dispatch(const DayanMoeSortConfig * config, const int32_t * selected, void * workspace) {
    if (config->sort.rows == 1) { return 0; }
    const uint32_t task = dayan_native_task_id();
    const uint32_t tokens = config->sort.rows, experts = config->sort.width;
    const auto layout = DayanMoeDispatchLayout::make(tokens, experts, config->experts_used);
    const uint64_t uram = (uintptr_t) dayan_uram_base();
    const uint64_t plane = ((dayan_uram_size() - 256) / 3) & ~255ULL;
    const uint64_t keys = uram + 256, ids = keys + plane, positions = ids + plane;
    constexpr long histogram = kMoeHistogram, reorder = kMoeReorder;
    const uint32_t row_bytes = config->experts_used * sizeof(int32_t);
    if (!workspace || !uram || layout.bytes > config->workspace_bytes || experts > kDayanSortMaxWidth ||
        5 * layout.count_stride + 512 > dayan_uram_size() ||
        !row_bytes || row_bytes > plane || (uint64_t) tokens * config->experts_used > INT32_MAX) {
        return -22;
    }
    int status = dayan_cluster_sync();
    if (status < 0) { return status; }

    moe_lane_ids(uram);
    const uint32_t local_tokens = ((uint64_t) tokens + 3U - task) / 4U;
    const uint32_t capacity = plane / row_bytes;
    for (uint32_t expert_first = 0; expert_first < experts; expert_first += kMoeExpertWindow) {
        const uint32_t count = experts - expert_first < kMoeExpertWindow ? experts - expert_first : kMoeExpertWindow;
        const uint32_t count_bytes = ((count + 63U) & ~63U) * sizeof(uint32_t);
        configure_vec_grid(count_bytes / 256);
        configure_vec_iterator<kVecV0Base>(0, 0, 256, 0);
        begin_vec_block(2);
        __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
        __tx_vec_lsu_v2sm<0, 0>(histogram);
        start_vec_block(2);
        dayan_wait_vec();
        for (uint32_t first = 0; first < local_tokens; first += capacity) {
            const uint32_t rows = local_tokens - first < capacity ? local_tokens - first : capacity;
            const uint32_t elements = rows * config->experts_used, blocks = (elements + 63) / 64;
            void * mapping = (char *) workspace + layout.positions_offset + task * layout.position_stride + (uint64_t) first * row_bytes;
            if (expert_first) { memory_load_bytes(positions, mapping, (uint64_t) rows * row_bytes); }
            sort_transfer<true>(keys, (uintptr_t) (selected + (uint64_t) (task + 4 * first) * config->sort.output_stride),
                                rows, row_bytes, row_bytes, 4ULL * config->sort.output_stride * sizeof(int32_t));
            configure_vec_grid(blocks);
            configure_vec_iterator<kVecV0Base>(0, 0, 256, 0);
            configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
            configure_vec_iterator<VEC_S0_BASE>(0, 0, 64, 0);
            begin_vec_block(19);
            __tx_vec_lsu_g2v_broadcast32<1, 2>(elements);
            __tx_vec_lsu_u2v<0, 2>(uram);
            __tx_vec_lsu_v2u<0, 0>(ids);
            __tx_vec_s_iter_gets<0, 0, 0, 0>();
            __tx_vec_lsu_s2v_broadcast32<0, 2, 0>();
            __tx_vec_vv_int32_add<0, 0, 1>();
            __tx_vec_vv_int32_cmp<0, 16, 0, 0b100>();
            __tx_vec_lsu_u2v<0, 0>(keys);
            __tx_vec_lsu_g2v_broadcast32<1, 2>(expert_first);
            __tx_vec_vv_int32_sub<0, 0, 16>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>(count);
            __tx_vec_vv_int32_cmp<0, 16, 1, 0b100>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>(0);
            __tx_vec_vv_int32_cmp<0, 16, 2, 0b011>();
            __tx_vec_s_bitwise<1, 1, 2, 0b1000>();
            __tx_vec_s_bitwise<0, 0, 1, 0b1000>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>(INT32_MAX);
            __tx_vec_vv_mux2<0, 16, 0, 0>();
            __tx_vec_lsu_v2u<0, 0>(keys);
            start_vec_block(19);
            dayan_wait_vec();

            sort_chunks(keys, ids, keys, ids, uram, 64, blocks, 256, 256, 64);
            configure_vec_grid(1);
            __tx_vec_tcsr_write<ggml_dayan_vec_detail::kVecDim1>(blocks - 1);
            configure_vec_iterator<kVecV0Base>(0, 256, 0, 0);
            configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
            begin_vec_block(62);
            __tx_vec_lsu_g2v_broadcast32<0, 2>(UINT32_MAX);
            __tx_vec_lsu_v2sm<0, 2>(reorder);
            __tx_vec_lsu_u2v<1, 0>(keys);
            __tx_vec_lsu_u2v<0, 2>(uram);
            __tx_vec_shuffle_ror<0, 16>(63);
            __tx_vec_vv_int32_cmp<0, 16, 0, 0b101>();
            __tx_vec_lsu_g2s<1>(1);
            __tx_vec_s_bitwise<0, 0, 1, 0b1110>();
            __tx_vec_lsu_g2v_broadcast32<0, 2>(0);
            __tx_vec_vv_mux2<0, 0, 2, 0>();
            moe_prefix_max();
            // Q1 retains expert IDs while Q0 computes the first lane of each run.
            __tx_vec_lsu_u2v<0, 2>(uram);
            __tx_vec_vv_int32_sub<0, 0, 1>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>(count);
            __tx_vec_vv_int32_cmp<23, 16, 2, 0b100>();
            __tx_vec_v_int32_sll<1, 23, 2>();
            __tx_vec_sm2v_sparse<0, 0, 16, 2>(histogram);
            __tx_vec_vv_int32_add<0, 0, 1>();
            __tx_vec_lsu_u2v<1, 0>(ids);
            __tx_vec_v_int32_sll<1, 16, 2>();
            __tx_vec_v2sm_sparse<0, 16, 2>(reorder);
            __tx_vec_shuffle_ror<0, 26>(1);
            __tx_vec_vv_int32_cmp<26, 0, 0, 0b101>();
            __tx_vec_lsu_g2s<1>(uint64_t{1} << 63);
            __tx_vec_s_bitwise<0, 0, 1, 0b1110>();
            __tx_vec_s_bitwise<0, 0, 2, 0b1000>();
            __tx_vec_lsu_g2v_broadcast32<0, 2>(1);
            __tx_vec_vv_int32_add<0, 0, 2>();
            // Only the final lane of each run writes its counter; addresses are unique.
            __tx_vec_v2sm_sparse<0, 18, 0>(histogram);
            __tx_vec_lsu_sm2v<0, 2>(reorder);
            __tx_vec_lsu_g2v_broadcast32<1, 2>(0);
            __tx_vec_vv_int32_cmp<0, 16, 3, 0b011>();
            __tx_vec_lsu_v2u_mask<0, 0, 3>(positions);
            // DIM1 keeps histogram updates ordered across blocks.
            normalization_start_loop(0, 62, 62);
            dayan_wait_vec();
            memory_store_bytes(mapping, positions, (uint64_t) rows * row_bytes);
        }
        configure_vec_grid(count_bytes / 256);
        configure_vec_iterator<kVecV0Base>(0, 0, 256, 0);
        begin_vec_block(2);
        __tx_vec_lsu_sm2v<0, 0>(histogram);
        __tx_vec_lsu_v2u<0, 0>(keys);
        start_vec_block(2);
        dayan_wait_vec();
        memory_store_bytes((char *) workspace + task * layout.count_stride + expert_first * sizeof(uint32_t), keys, count_bytes);
    }
    status = dayan_cluster_sync();
    if (status < 0) { return status; }
    if (task == 0) {
        // Scan expert totals and add each core's exclusive count prefix.
        const uint64_t offsets = keys + 4 * layout.count_stride, carry = offsets + layout.count_stride;
        memory_load_bytes(keys, workspace, 4 * layout.count_stride);
        configure_vec_grid(1);
        __tx_vec_tcsr_write<ggml_dayan_vec_detail::kVecDim1>(layout.count_stride / 256 - 1);
        configure_vec_iterator<kVecV0Base>(0, 256, 0, 0);
        configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
        begin_vec_block(56);
        __tx_vec_lsu_g2v_broadcast32<1, 2>(0);
        __tx_vec_lsu_u2v<0, 0>(keys);
        __tx_vec_lsu_u2v<0, 0>(keys + layout.count_stride);
        __tx_vec_vv_int32_add<0, 0, 1>();
        __tx_vec_lsu_u2v<0, 0>(keys + 2 * layout.count_stride);
        __tx_vec_vv_int32_add<0, 0, 1>();
        __tx_vec_lsu_u2v<0, 0>(keys + 3 * layout.count_stride);
        __tx_vec_vv_int32_add<0, 0, 1>();
        __tx_vec_lsu_g2v_broadcast32<0, 2>(0);
        __tx_vec_lsu_v2u<0, 0>(keys);
        __tx_vec_lsu_v2u<7, 0>(keys + layout.count_stride);
        __tx_vec_lsu_v2u<5, 0>(keys + 2 * layout.count_stride);
        __tx_vec_lsu_v2u<3, 0>(keys + 3 * layout.count_stride);
        __tx_vec_shuffle_xor<0, 1>(0);
        moe_prefix_sum();
        __tx_vec_vv_int32_add<0, 0, 22>();
        __tx_vec_vv_int32_sub<0, 0, 13>();
        __tx_vec_lsu_v2u<0, 0>(offsets);
        __tx_vec_shuffle_xor<1, 0>(0);
        for (uint32_t core = 0; core < 4; ++core) {
            __tx_vec_lsu_u2v<0, 0>(keys + core * layout.count_stride);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_lsu_v2u<0, 0>(keys + core * layout.count_stride);
        }
        __tx_vec_shuffle_broadcast<1, 9>(63);
        __tx_vec_lsu_v2u<16, 2>(carry);
        normalization_start_loop(1, 55, 56);
        dayan_wait_vec();
        memory_store_bytes(workspace, keys, 4 * layout.count_stride);
        memory_store_bytes((char *) workspace + layout.offsets_offset, offsets, experts * sizeof(uint32_t));
        memory_store_bytes((char *) workspace + layout.offsets_offset + experts * sizeof(uint32_t), carry, sizeof(uint32_t));
    }
    status = dayan_cluster_sync();
    if (status < 0) { return status; }
    for (uint32_t expert_first = 0; expert_first < experts; expert_first += kMoeExpertWindow) {
        const uint32_t count = experts - expert_first < kMoeExpertWindow ? experts - expert_first : kMoeExpertWindow;
        const uint32_t count_bytes = ((count + 63U) & ~63U) * sizeof(uint32_t);
        memory_load_bytes(keys, (char *) workspace + task * layout.count_stride + expert_first * sizeof(uint32_t), count_bytes);
        configure_vec_grid(count_bytes / 256);
        configure_vec_iterator<kVecV0Base>(0, 0, 256, 0);
        begin_vec_block(2);
        __tx_vec_lsu_u2v<0, 0>(keys);
        __tx_vec_lsu_v2sm<0, 0>(histogram);
        start_vec_block(2);
        dayan_wait_vec();
        for (uint32_t first = 0; first < local_tokens; first += capacity) {
            const uint32_t rows = local_tokens - first < capacity ? local_tokens - first : capacity;
            const uint32_t elements = rows * config->experts_used;
            void * mapping = (char *) workspace + layout.positions_offset + task * layout.position_stride + (uint64_t) first * row_bytes;
            sort_transfer<true>(keys, (uintptr_t) (selected + (uint64_t) (task + 4 * first) * config->sort.output_stride),
                                rows, row_bytes, row_bytes, 4ULL * config->sort.output_stride * sizeof(int32_t));
            memory_load_bytes(positions, mapping, (uint64_t) elements * sizeof(uint32_t));
            configure_vec_grid((elements + 63) / 64);
            configure_vec_iterator<kVecV0Base>(0, 0, 256, 0);
            configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
            configure_vec_iterator<VEC_S0_BASE>(0, 0, 64, 0);
            begin_vec_block(20);
            __tx_vec_lsu_u2v<0, 2>(uram);
            __tx_vec_s_iter_gets<0, 0, 0, 0>();
            __tx_vec_lsu_s2v_broadcast32<1, 2, 0>();
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>(elements);
            __tx_vec_vv_int32_cmp<0, 16, 2, 0b100>();
            __tx_vec_lsu_u2v<0, 0>(keys);
            __tx_vec_lsu_g2v_broadcast32<1, 2>(expert_first);
            __tx_vec_vv_int32_sub<0, 0, 16>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>(count);
            __tx_vec_vv_int32_cmp<0, 16, 0, 0b100>();
            __tx_vec_lsu_g2v_broadcast32<1, 2>(0);
            __tx_vec_vv_int32_cmp<0, 16, 1, 0b011>();
            __tx_vec_s_bitwise<0, 0, 1, 0b1000>();
            __tx_vec_s_bitwise<2, 2, 0, 0b1000>();
            __tx_vec_v_int32_sll<0, 0, 2>();
            __tx_vec_sm2v_sparse<0, 0, 0, 2>(histogram);
            __tx_vec_lsu_u2v<1, 0>(positions);
            __tx_vec_vv_int32_add<0, 0, 16>();
            __tx_vec_lsu_v2u_mask<0, 0, 2>(positions);
            start_vec_block(20);
            dayan_wait_vec();
            memory_store_bytes(mapping, positions, (uint64_t) elements * sizeof(uint32_t));
        }
    }
    return 0;
}

static inline int run_moe_sort(const DayanMoeSortConfig * config, const float * probabilities, int32_t * selected, void * workspace) {
    if (!config || !probabilities || !selected || !config->sort.rows || !config->sort.width ||
        !config->experts_used || config->experts_used > config->sort.output_width ||
        config->sort.order != kDayanSortOrderDesc || dayan_native_task_count() != 4) { return -22; }
    run_sort<false>(&config->sort, probabilities, selected);
    return run_moe_dispatch(config, selected, workspace);
}

static inline void moe_address_table(uint64_t table, uint64_t rows, uint64_t base, uint32_t stride) {
    // Expand 32 row IDs to low/high address pairs; propagate the low-word carry.
    begin_vec_block(15);
    __tx_vec_lsu_u2v<0, 0>(rows);
    __tx_vec_shuffle_compress<0, 0>(0x20);
    __tx_vec_lsu_g2v_broadcast32<1, 2>(stride);
    __tx_vec_vv_int32_mulu24<0, 0, 16>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>((uint32_t) base);
    __tx_vec_vv_int32_add<0, 0, 16>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>(0x80000000U);
    __tx_vec_vv_int32_bitwise<0, 0, 16, 0b0110>();
    __tx_vec_vv_int32_bitwise<1, 17, 16, 0b0110>();
    __tx_vec_vv_int32_cmp<0, 16, 0, 0b100>();
    __tx_vec_lsu_g2v_broadcast32<1, 2>(base >> 32);
    __tx_vec_lsu_g2v_mask_broadcast32<1, 16, 0>((base >> 32) + 1);
    __tx_vec_lsu_g2s<1>(0xaaaaaaaaaaaaaaaaULL);
    __tx_vec_vv_mux2<0, 1, 16, 1>();
    __tx_vec_lsu_v2u<0, 1>(table);
    start_vec_block(15);
}

static void moe_permute_rows(const DayanMxfp4MoeConfig * config, void * workspace, void * linear, void * linear_scales,
                             void * grouped, void * grouped_scales, bool reverse) {
    const auto & r = config->routing;
    const auto & mm = config->matmul;
    const uint32_t task = dayan_native_task_id(), slots = reverse ? r.experts_used : r.activation_slots;
    const uint32_t local_tokens = ((uint64_t) r.tokens + 3U - task) / 4U;
    if (!local_tokens) { return; }
    const uint32_t total = r.tokens * r.experts_used, elements = local_tokens * r.experts_used;
    const uint32_t columns = reverse ? mm.n : mm.k;
    const bool mxfp4 = !reverse || mm.output_type == kDayanDataTypeMXFP4;
    const auto layout = DayanMxfp4Layout::make(total, columns);
    const auto dispatch = DayanMoeDispatchLayout::make(r.tokens, r.experts, r.experts_used);
    const auto * mapping = (const uint32_t *) ((char *) workspace + dispatch.positions_offset + task * dispatch.position_stride);
    const uint64_t linear_stride = reverse ? mm.output_stride : mm.a_stride;
    const uint64_t linear_scale_stride = reverse ? mm.output_scale_stride : mm.a_scale_stride;
    const uint64_t grouped_stride = mxfp4 ? layout.row_stride : columns * sizeof(float);
    const uint64_t max_stride = linear_stride > grouped_stride ? linear_stride : grouped_stride;
    const uint32_t planes = (columns + 127U) / 128U;
    if ((r.experts_used & (r.experts_used - 1)) || total >= (1U << 24) || max_stride >= (1U << 24) ||
        (uint64_t) (total - 1) * max_stride > UINT32_MAX || !memory_stride_fits(linear_scale_stride) ||
        !memory_stride_fits(layout.scale_stride)) {
        // Large offsets use cached DRAM metadata; no scalar URAM access is needed.
        dayan_cache_invalidate(mapping, (uint64_t) elements * sizeof(uint32_t));
        for (uint32_t index = 0; index < elements; ++index) {
            const uint32_t row = (task + 4 * (index / r.experts_used)) * slots + index % slots;
            const uint32_t group = mapping[index];
            void * natural = (char *) linear + (uint64_t) row * linear_stride;
            void * ordered = (char *) grouped + (uint64_t) group * grouped_stride;
            memory_copy_device(reverse ? natural : ordered, reverse ? ordered : natural, grouped_stride);
            if (!mxfp4) { continue; }
            for (uint32_t plane = 0; plane < planes; ++plane) {
                natural = (char *) linear_scales + (uint64_t) plane * linear_scale_stride + (uint64_t) row * 4;
                ordered = (char *) grouped_scales + (uint64_t) plane * layout.scale_stride + (uint64_t) group * 4;
                memory_copy_device(reverse ? natural : ordered, reverse ? ordered : natural, 4);
            }
        }
        return;
    }
    const uint64_t uram = (uintptr_t) dayan_uram_base();
    const uint32_t capacity = ((dayan_uram_size() - 512) / 24) & ~63U;
    const uint64_t group_rows = uram + 256, natural_rows = group_rows + (uint64_t) capacity * 4 + 256;
    const uint64_t natural_table = natural_rows + (uint64_t) capacity * 4, group_table = natural_table + (uint64_t) capacity * 8;
    moe_lane_ids(uram);
    for (uint32_t first = 0; first < elements; first += capacity) {
        const uint32_t count = elements - first < capacity ? elements - first : capacity;
        memory_load_bytes(group_rows, mapping + first, (uint64_t) count * 4);
        configure_vec_grid((count + 63) / 64);
        configure_vec_iterator<kVecV0Base>(0, 0, 256, 0);
        configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
        configure_vec_iterator<VEC_S0_BASE>(0, 0, 64, 0);
        __tx_vec_tcsr_write<VEC_S0_BASE>(first);
        begin_vec_block(14);
        __tx_vec_lsu_u2v<0, 2>(uram);
        __tx_vec_s_iter_gets<0, 0, 0, 0>();
        __tx_vec_lsu_s2v_broadcast32<1, 2, 0>();
        __tx_vec_vv_int32_add<0, 0, 16>();
        __tx_vec_lsu_g2v_broadcast32<1, 2>(__builtin_ctz(r.experts_used));
        __tx_vec_vv_int32_srl<0, 0, 16>();
        __tx_vec_lsu_g2v_broadcast32<1, 2>(4 * slots);
        __tx_vec_vv_int32_mulu24<0, 0, 16>();
        __tx_vec_lsu_g2v_broadcast32<1, 2>(slots - 1);
        __tx_vec_vv_int32_bitwise<0, 2, 16, 0b1000>();
        __tx_vec_vv_int32_add<0, 0, 1>();
        __tx_vec_lsu_g2v_broadcast32<1, 2>(task * slots);
        __tx_vec_vv_int32_add<0, 0, 16>();
        __tx_vec_lsu_v2u<0, 0>(natural_rows);
        start_vec_block(14);
        dayan_wait_vec();
        configure_vec_grid((count + 31) / 32);
        configure_vec_iterator<kVecV0Base>(0, 0, 128, 0);
        configure_vec_iterator<kVecV1Base>(0, 0, 256, 0);
        moe_address_table(natural_table, natural_rows, (uintptr_t) linear, linear_stride);
        moe_address_table(group_table, group_rows, (uintptr_t) grouped, grouped_stride);
        dayan_wait_vec();
        configure_memory_d2d_lsu<1>(grouped_stride / 64, 64, 64, 64);
        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_TABLESIZE>(count - 1);
        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_TABLESTRIDE>(8);
        __tx_lsu_dma_d2d_sg<1>(reverse ? natural_table : group_table, reverse ? group_table : natural_table);
        dayan_wait_lsu();
        if (!mxfp4) { continue; }
        moe_address_table(natural_table, natural_rows, (uintptr_t) linear_scales, 4);
        moe_address_table(group_table, group_rows, (uintptr_t) grouped_scales, 4);
        dayan_wait_vec();
        configure_memory_d2d_lsu<1>(planes, reverse ? layout.scale_stride : linear_scale_stride,
                                   reverse ? linear_scale_stride : layout.scale_stride, 4);
        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_TABLESIZE>(count - 1);
        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_TABLESTRIDE>(8);
        __tx_lsu_dma_d2d_sg<1>(reverse ? natural_table : group_table, reverse ? group_table : natural_table);
        dayan_wait_lsu();
    }
}

static int run_mul_mat_id_mxfp4_grouped(const DayanMxfp4MoeConfig * config, const void * a, const void * as,
                                         const void * b, const void * bs, const void * up, const void * ups,
                                         const int32_t * ids, void * output, void * scales, void * workspace) {
    if (!config) {
        return -22;
    }
    const auto & r = config->routing;
    const auto & mm = config->matmul;
    if (!r.experts_used || r.tokens > UINT32_MAX / r.experts_used || r.experts > kDayanSortMaxWidth) {
        return -22;
    }
    const uint32_t total = r.tokens * r.experts_used;
    const auto dispatch = DayanMoeDispatchLayout::make(r.tokens, r.experts, r.experts_used);
    const DayanMxfp4Layout input_layout = DayanMxfp4Layout::make(total, mm.k);
    const DayanMxfp4Layout output_layout = DayanMxfp4Layout::make(total, mm.n);
    const uint64_t output_bytes = mm.output_type == kDayanDataTypeMXFP4 ? output_layout.bytes :
                                                                            (uint64_t) total * mm.n * sizeof(float);
    const uint64_t output_offset = (input_layout.bytes + 63U) & ~63ULL;
    const uint64_t required = dispatch.bytes + output_offset + ((output_bytes + 63U) & ~63ULL);
    if (!config || !workspace || !a || !as || !b || !bs || !up || !ups || !ids || !output ||
        !r.tokens || !r.experts_used || !r.experts || !r.activation_slots || !config->workspace_bytes ||
        required > config->workspace_bytes || r.experts_used % r.activation_slots || dayan_native_task_count() != 4 ||
        mm.m != 1 || mm.weight_type != kDayanDataTypeMXFP4 ||
        (mm.output_type == kDayanDataTypeMXFP4 && !scales)) {
        return -22;
    }
    const uint32_t * offsets = (const uint32_t *) ((const char *) workspace + dispatch.offsets_offset);
    dayan_cache_invalidate(offsets, ((uint64_t) r.experts + 1) * sizeof(uint32_t));
    if (offsets[r.experts] != total) { return -22; }
    const uint64_t grouped_a = (uintptr_t) workspace + dispatch.bytes;
    const uint64_t grouped_as = grouped_a + input_layout.scale_offset;
    const uint64_t grouped_output = grouped_a + output_offset;
    const uint64_t grouped_output_scales = grouped_output + output_layout.scale_offset;
    moe_permute_rows(config, workspace, (void *) a, (void *) as, (void *) grouped_a, (void *) grouped_as, false);
    int status = dayan_cluster_sync();
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
    status = dayan_cluster_sync();
    if (status < 0) { return status; }
    moe_permute_rows(config, workspace, output, scales, (void *) grouped_output, (void *) grouped_output_scales, true);
    return dayan_cluster_sync();
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
        r.ids_row_stride > UINT64_MAX / r.tokens || mm.m != 1 || mm.n != r.n || mm.k != r.k || mm.weight_type != kDayanDataTypeMXFP4) { return -22; }
    dayan_cache_invalidate(ids, (uint64_t) (r.tokens - 1) * r.ids_row_stride + r.experts_used * sizeof(int32_t));
    ggml_dayan_mxfp4_detail::Phases phases;
    for (uint32_t token = 0; token < r.tokens; ++token) {
        const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * r.ids_row_stride);
        for (uint32_t used = 0; used < r.experts_used; ++used) {
            const int32_t expert = row_ids[used];
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
    dayan_cache_invalidate(ids, (uint64_t) (config->tokens - 1) * config->ids_row_stride + config->experts_used * sizeof(int32_t));
    const uint64_t weight_elements = (uint64_t) config->n * config->k;
    MatmulPhases local_phases;
    MatmulPhases & resource_phases = phases ? *phases : local_phases;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * config->ids_row_stride);
            const int32_t expert = row_ids[used];
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
    dayan_cache_invalidate(ids, (uint64_t) (config->tokens - 1) * config->ids_row_stride + config->experts_used * sizeof(int32_t));
    const uint64_t weight_elements = (uint64_t) config->n * config->k;
    MatmulPhases resource_phases;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * config->ids_row_stride);
            const int32_t expert = row_ids[used];
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
                              float *                  output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {(uint64_t) config->width * config->experts_used * config->tokens, (uint64_t) config->experts_used * config->tokens, 256U * 1024U}, [&](const NativeContext & worker) {
            run_add_id(config, input, bias, ids, output, worker);
        })) { return; }

    if (config == nullptr || input == nullptr || bias == nullptr || ids == nullptr || output == nullptr ||
        config->width == 0 || config->experts == 0 || config->experts > INT32_MAX || config->experts_used == 0 ||
        config->tokens == 0 || config->ids_row_stride < (uint64_t) config->experts_used * sizeof(int32_t) ||
        config->ids_row_stride > UINT64_MAX / config->tokens ||
        !moe_allocation_fits(sizeof(float), config->width, config->experts_used, config->tokens) ||
        !moe_allocation_fits(sizeof(float), config->width, config->experts, 1) ||
        !moe_allocation_fits(sizeof(int32_t), config->experts_used, config->tokens, 1)) {
        DAYAN_PARAMS_ERROR();
    }
    dayan_cache_invalidate(ids, (uint64_t) (config->tokens - 1) * config->ids_row_stride + config->experts_used * sizeof(int32_t));
    elementwise_validate((uint64_t) config->width * config->experts_used * config->tokens, input, output, context);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = native_uram(context);
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t bias_uram  = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id(context);
    const uint32_t tasks      = dayan_native_task_count(context);
    const uint64_t rows       = (uint64_t) config->tokens * config->experts_used;
    for (uint64_t row = task; row < rows; row += tasks) {
        const uint32_t used   = (uint32_t) (row % config->experts_used);
        const uint32_t token  = (uint32_t) (row / config->experts_used);
        const int32_t * token_ids =
            (const int32_t *) ((const char *) ids + (uint64_t) token * config->ids_row_stride);
        const int32_t expert = token_ids[used];
        if (expert < 0 || expert >= (int32_t) config->experts) {
            DAYAN_PARAMS_ERROR();
        }
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input + row * config->width + offset, elements, context);
            normalization_load(bias_uram, bias + (uint64_t) expert * config->width + offset, elements, context);
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

}  // namespace ggml_dayan_matmul_detail

#endif
