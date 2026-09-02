#pragma once

#include "ggml-dayan-elementwise.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static inline uint32_t memory_type_bytes(uint32_t type) {
    if (type == kDayanDataTypeF32 || type == kDayanDataTypeI32) {
        return sizeof(uint32_t);
    }
    return type == kDayanDataTypeI64 ? sizeof(uint64_t) : sizeof(uint16_t);
}

enum : uint64_t {
    kMemoryLsuMaxDimension = 1ULL << 24,
    kMemoryLsuMaxStride    = (1ULL << 28) - 1U,
    kMemoryMaxCopyDimensions = 8,
};

struct MemoryCopyDimension {
    uint64_t count;
    uint64_t input_stride;
    uint64_t output_stride;
};

template <long Group>
static inline void configure_memory_lsu_3d(uint32_t dim0,
                                           uint64_t input_dim0_stride,
                                           uint64_t output_dim0_stride,
                                           uint32_t dim1,
                                           uint64_t input_dim1_stride,
                                           uint64_t output_dim1_stride,
                                           uint32_t dim2,
                                           uint64_t input_dim2_stride,
                                           uint64_t output_dim2_stride,
                                           uint32_t read_bytes,
                                           uint32_t write_bytes) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(dim0 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(dim1 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(dim2 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(input_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(input_dim2_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(output_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(output_dim2_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(read_bytes - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(write_bytes - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group>
static inline void configure_memory_lsu(uint32_t transactions,
                                        uint64_t input_stride,
                                        uint64_t output_stride,
                                        uint32_t read_bytes,
                                        uint32_t write_bytes) {
    configure_memory_lsu_3d<Group>(transactions, input_stride, output_stride, 1, 0, 0, 1, 0, 0, read_bytes,
                                   write_bytes);
}

template <long Group>
static inline void configure_memory_d2d_lsu(uint32_t transactions,
                                            uint64_t input_stride,
                                            uint64_t output_stride,
                                            uint32_t transaction_bytes) {
    configure_memory_lsu<Group>(transactions, input_stride, output_stride, transaction_bytes, transaction_bytes);
}

static inline bool memory_stride_fits(uint64_t stride) {
    return stride <= kMemoryLsuMaxStride;
}

static inline bool memory_dimensions_contiguous(const MemoryCopyDimension & inner,
                                                const MemoryCopyDimension & outer) {
    if (inner.input_stride != 0 && inner.count > UINT64_MAX / inner.input_stride) {
        return false;
    }
    if (inner.output_stride != 0 && inner.count > UINT64_MAX / inner.output_stride) {
        return false;
    }
    return outer.input_stride == inner.input_stride * inner.count &&
           outer.output_stride == inner.output_stride * inner.count;
}

static inline uint32_t memory_compact_dimensions(MemoryCopyDimension * dimensions, uint32_t count) {
    uint32_t compact_count = 0;
    for (uint32_t index = 0; index < count; ++index) {
        if (dimensions[index].count <= 1U) {
            continue;
        }
        if (compact_count != 0U && memory_dimensions_contiguous(dimensions[compact_count - 1U], dimensions[index]) &&
            dimensions[compact_count - 1U].count <= UINT64_MAX / dimensions[index].count) {
            dimensions[compact_count - 1U].count *= dimensions[index].count;
            continue;
        }
        dimensions[compact_count++] = dimensions[index];
    }
    return compact_count;
}

template <bool SourceUram>
static inline void memory_issue_3d(uint64_t destination,
                                   uint64_t source,
                                   uint32_t transaction_bytes,
                                   const MemoryCopyDimension dimensions[3]) {
    for (uint64_t first2 = 0; first2 < dimensions[2].count;) {
        const uint32_t count2 = dimensions[2].count - first2 < kMemoryLsuMaxDimension ?
                                    (uint32_t) (dimensions[2].count - first2) :
                                    (uint32_t) kMemoryLsuMaxDimension;
        for (uint64_t first1 = 0; first1 < dimensions[1].count;) {
            const uint32_t count1 = dimensions[1].count - first1 < kMemoryLsuMaxDimension ?
                                        (uint32_t) (dimensions[1].count - first1) :
                                        (uint32_t) kMemoryLsuMaxDimension;
            for (uint64_t first0 = 0; first0 < dimensions[0].count;) {
                const uint32_t count0 = dimensions[0].count - first0 < kMemoryLsuMaxDimension ?
                                            (uint32_t) (dimensions[0].count - first0) :
                                            (uint32_t) kMemoryLsuMaxDimension;
                configure_memory_lsu_3d<1>(count0, dimensions[0].input_stride, dimensions[0].output_stride, count1,
                                           dimensions[1].input_stride, dimensions[1].output_stride, count2,
                                           dimensions[2].input_stride, dimensions[2].output_stride, transaction_bytes,
                                           transaction_bytes);
                const uint64_t output_address = destination + first0 * dimensions[0].output_stride +
                                                first1 * dimensions[1].output_stride +
                                                first2 * dimensions[2].output_stride;
                const uint64_t input_address = source + first0 * dimensions[0].input_stride +
                                               first1 * dimensions[1].input_stride +
                                               first2 * dimensions[2].input_stride;
                if constexpr (SourceUram) {
                    __tx_lsu_dma_u2d<1>(output_address, input_address);
                } else {
                    __tx_lsu_dma_d2d<1>(output_address, input_address);
                }
                first0 += count0;
            }
            first1 += count1;
        }
        first2 += count2;
    }
}

template <bool SourceUram>
static inline void memory_issue_region_from(uint64_t destination,
                                            uint64_t source,
                                            uint32_t transaction_bytes,
                                            MemoryCopyDimension * dimensions,
                                            uint32_t dimension_count) {
    if (dimension_count > kMemoryMaxCopyDimensions) {
        DAYAN_PARAMS_ERROR();
    }
    dimension_count = memory_compact_dimensions(dimensions, dimension_count);

    MemoryCopyDimension encoded[3] = { { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 } };
    MemoryCopyDimension looped[kMemoryMaxCopyDimensions] = {};
    bool                selected[kMemoryMaxCopyDimensions] = {};
    uint32_t            encoded_count = 0;
    uint32_t            looped_count  = 0;

    for (; encoded_count < 3U; ++encoded_count) {
        int best = -1;
        for (uint32_t index = 0; index < dimension_count; ++index) {
            if (!selected[index] && memory_stride_fits(dimensions[index].input_stride) &&
                memory_stride_fits(dimensions[index].output_stride) &&
                (best < 0 || dimensions[index].count > dimensions[best].count)) {
                best = (int) index;
            }
        }
        if (best < 0) {
            break;
        }
        selected[best]          = true;
        encoded[encoded_count] = dimensions[best];
    }
    for (uint32_t index = 0; index < dimension_count; ++index) {
        if (!selected[index]) {
            looped[looped_count++] = dimensions[index];
        }
    }

    uint64_t combinations = 1;
    for (uint32_t index = 0; index < looped_count; ++index) {
        if (looped[index].count == 0 || combinations > UINT64_MAX / looped[index].count) {
            DAYAN_PARAMS_ERROR();
        }
        combinations *= looped[index].count;
    }
    for (uint64_t linear = 0; linear < combinations; ++linear) {
        uint64_t index         = linear;
        uint64_t input_offset  = 0;
        uint64_t output_offset = 0;
        for (uint32_t dimension = 0; dimension < looped_count; ++dimension) {
            const uint64_t coordinate = index % looped[dimension].count;
            index /= looped[dimension].count;
            input_offset += coordinate * looped[dimension].input_stride;
            output_offset += coordinate * looped[dimension].output_stride;
        }
        memory_issue_3d<SourceUram>(destination + output_offset, source + input_offset, transaction_bytes, encoded);
    }
}

static inline void memory_issue_region(uint64_t destination,
                                       uint64_t source,
                                       uint32_t transaction_bytes,
                                       MemoryCopyDimension * dimensions,
                                       uint32_t dimension_count) {
    memory_issue_region_from<false>(destination, source, transaction_bytes, dimensions, dimension_count);
}

static inline void memory_issue_copy_dimensions(uint64_t destination,
                                                uint64_t source,
                                                uint64_t row_bytes,
                                                MemoryCopyDimension * dimensions,
                                                uint32_t dimension_count) {
    if (dimension_count > kMemoryMaxCopyDimensions ||
        (row_bytes > 64U && dimension_count == kMemoryMaxCopyDimensions)) {
        DAYAN_PARAMS_ERROR();
    }
    if (row_bytes <= 64U) {
        memory_issue_region(destination, source, (uint32_t) row_bytes, dimensions, dimension_count);
        return;
    }

    const uint64_t full_blocks = row_bytes / 64U;
    const uint32_t tail        = (uint32_t) (row_bytes % 64U);
    if (full_blocks != 0U) {
        MemoryCopyDimension full[kMemoryMaxCopyDimensions] = {};
        full[0] = { full_blocks, 64, 64 };
        for (uint32_t index = 0; index < dimension_count; ++index) {
            full[index + 1U] = dimensions[index];
        }
        memory_issue_region(destination, source, 64, full, dimension_count + 1U);
    }
    if (tail != 0U) {
        const uint64_t offset = full_blocks * 64U;
        memory_issue_region(destination + offset, source + offset, tail, dimensions, dimension_count);
    }
}

static inline void memory_issue_fill_dimensions(uint64_t destination,
                                                uint64_t source,
                                                uint64_t row_bytes,
                                                const MemoryCopyDimension * dimensions,
                                                uint32_t dimension_count) {
    if (dimension_count > kMemoryMaxCopyDimensions ||
        (row_bytes > 64U && dimension_count == kMemoryMaxCopyDimensions)) {
        DAYAN_PARAMS_ERROR();
    }

    MemoryCopyDimension outer[kMemoryMaxCopyDimensions] = {};
    for (uint32_t index = 0; index < dimension_count; ++index) {
        outer[index] = { dimensions[index].count, 0, dimensions[index].output_stride };
    }
    if (row_bytes <= 64U) {
        memory_issue_region_from<true>(destination, source, (uint32_t) row_bytes, outer, dimension_count);
        return;
    }

    const uint64_t full_blocks = row_bytes / 64U;
    const uint32_t tail        = (uint32_t) (row_bytes % 64U);
    if (full_blocks != 0U) {
        MemoryCopyDimension full[kMemoryMaxCopyDimensions] = {};
        full[0] = { full_blocks, 0, 64 };
        for (uint32_t index = 0; index < dimension_count; ++index) {
            full[index + 1U] = outer[index];
        }
        memory_issue_region_from<true>(destination, source, 64, full, dimension_count + 1U);
    }
    if (tail != 0U) {
        memory_issue_region_from<true>(destination + full_blocks * 64U, source, tail, outer, dimension_count);
    }
}

static inline void memory_issue_fill(uint64_t destination, uint64_t source, uint64_t bytes) {
    memory_issue_fill_dimensions(destination, source, bytes, nullptr, 0);
}

static inline void memory_issue_copy_rows(uint64_t destination,
                                          uint64_t source,
                                          uint64_t row_bytes,
                                          const uint64_t counts[3],
                                          const uint64_t input_strides[3],
                                          const uint64_t output_strides[3]) {
    MemoryCopyDimension dimensions[3] = {
        { counts[0], input_strides[0], output_strides[0] },
        { counts[1], input_strides[1], output_strides[1] },
        { counts[2], input_strides[2], output_strides[2] },
    };
    memory_issue_copy_dimensions(destination, source, row_bytes, dimensions, 3);
}

static inline bool memory_issue_linear_copy(uint64_t destination,
                                            uint64_t source,
                                            uint64_t bytes) {
    const uint64_t full_blocks = bytes / 64U;
    bool           issued      = false;

    if (full_blocks != 0U) {
        MemoryCopyDimension dimensions[4] = {
            { full_blocks, 64, 64 },
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 },
        };
        memory_issue_region(destination, source, 64, dimensions, 1);
        issued = true;
    }
    const uint32_t tail = (uint32_t) (bytes % 64U);
    if (tail != 0U) {
        MemoryCopyDimension dimensions[4] = {
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 },
        };
        memory_issue_region(destination + full_blocks * 64U, source + full_blocks * 64U, tail, dimensions, 0);
        issued = true;
    }
    return issued;
}

static inline bool memory_issue_tensor_copy(uint64_t destination,
                                            uint64_t source,
                                            const uint32_t ne[4],
                                            const uint64_t input_nb[4],
                                            const uint64_t output_nb[4],
                                            uint32_t element_bytes) {
    for (uint32_t dimension = 0; dimension < 4U; ++dimension) {
        if (ne[dimension] == 0U) {
            return false;
        }
    }
    uint64_t linear_bytes = element_bytes;
    uint32_t first_outer  = 0;
    for (; first_outer < 4U; ++first_outer) {
        if (ne[first_outer] <= 1U) {
            continue;
        }
        if (input_nb[first_outer] != linear_bytes || output_nb[first_outer] != linear_bytes) {
            break;
        }
        linear_bytes *= ne[first_outer];
    }
    bool remaining_unit = true;
    for (uint32_t dimension = first_outer; dimension < 4U; ++dimension) {
        remaining_unit = remaining_unit && ne[dimension] == 1U;
    }
    if (remaining_unit) {
        return memory_issue_linear_copy(destination, source, linear_bytes);
    }

    MemoryCopyDimension dimensions[4] = {
        { 1, 0, 0 },
        { 1, 0, 0 },
        { 1, 0, 0 },
        { 1, 0, 0 },
    };
    uint32_t dimension_count = 0;
    uint64_t row_bytes       = linear_bytes;
    if (row_bytes == element_bytes && first_outer == 0U) {
        dimensions[dimension_count++] = { ne[0], input_nb[0], output_nb[0] };
        first_outer = 1U;
    }
    for (uint32_t dimension = first_outer; dimension < 4U; ++dimension) {
        dimensions[dimension_count++] = { ne[dimension], input_nb[dimension], output_nb[dimension] };
    }
    dimension_count = memory_compact_dimensions(dimensions, dimension_count);

    if (row_bytes == element_bytes) {
        memory_issue_region(destination, source, (uint32_t) row_bytes, dimensions, dimension_count);
        return true;
    }

    uint64_t counts[3]         = { 1, 1, 1 };
    uint64_t input_strides[3]  = { 0, 0, 0 };
    uint64_t output_strides[3] = { 0, 0, 0 };
    for (uint32_t index = 0; index < dimension_count; ++index) {
        counts[index]         = dimensions[index].count;
        input_strides[index]  = dimensions[index].input_stride;
        output_strides[index] = dimensions[index].output_stride;
    }
    memory_issue_copy_rows(destination, source, row_bytes, counts, input_strides, output_strides);
    return true;
}

static inline void memory_validate(const void * config) {
    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    (void) dayan_native_task_id();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_thread_exit(-12);
    }
}

static inline uint64_t memory_tile_count(uint64_t elements) {
    return (elements + NormalizationLayout::kTileElements - 1U) / NormalizationLayout::kTileElements;
}

static inline void memory_copy_uram(uint64_t destination, uint64_t source, uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    bool           issued = false;
    if (full != 0U) {
        configure_memory_d2d_lsu<1>(full, 64, 64, 64);
        __tx_lsu_dma_u2u<1>((long) destination, (long) source);
        issued = true;
    }
    if (tail != 0U) {
        configure_memory_d2d_lsu<1>(1, 0, 0, tail);
        __tx_lsu_dma_u2u<1>((long) (destination + (uint64_t) full * 64U),
                            (long) (source + (uint64_t) full * 64U));
        issued = true;
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline void memory_load_bytes(uint64_t destination, const void * source, uint64_t bytes) {
    constexpr uint64_t max_bytes = kMemoryLsuMaxDimension * 64U;
    uint64_t           offset    = 0;
    bool               issued    = false;
    while (bytes - offset >= 64U) {
        const uint64_t remaining = bytes - offset;
        const uint64_t chunk     = remaining < max_bytes ? remaining & ~63ULL : max_bytes;
        configure_memory_lsu<1>((uint32_t) (chunk / 64U), 64, 64, 64, 64);
        __tx_lsu_dma_d2u<1>((long) (destination + offset), (long) ((const char *) source + offset));
        issued = true;
        offset += chunk;
    }
    if (offset != bytes) {
        const uint32_t tail = (uint32_t) (bytes - offset);
        configure_memory_lsu<1>(1, 0, 0, tail, tail);
        __tx_lsu_dma_d2u<1>((long) (destination + offset), (long) ((const char *) source + offset));
        issued = true;
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline void memory_store_bytes(void * destination, uint64_t source, uint64_t bytes) {
    constexpr uint64_t max_bytes = kMemoryLsuMaxDimension * 64U;
    uint64_t           offset    = 0;
    bool               issued    = false;
    while (bytes - offset >= 64U) {
        const uint64_t remaining = bytes - offset;
        const uint64_t chunk     = remaining < max_bytes ? remaining & ~63ULL : max_bytes;
        configure_memory_lsu<1>((uint32_t) (chunk / 64U), 64, 64, 64, 64);
        __tx_lsu_dma_u2d<1>((long) ((char *) destination + offset), (long) (source + offset));
        issued = true;
        offset += chunk;
    }
    if (offset != bytes) {
        const uint32_t tail = (uint32_t) (bytes - offset);
        configure_memory_lsu<1>(1, 0, 0, tail, tail);
        __tx_lsu_dma_u2d<1>((long) ((char *) destination + offset), (long) (source + offset));
        issued = true;
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline void memory_load(uint64_t uram, const void * dram, uint32_t elements, uint32_t type) {
    memory_load_bytes(uram, dram, (uint64_t) elements * memory_type_bytes(type));
}

static inline void memory_store(void * dram, uint64_t uram, uint32_t elements, uint32_t type) {
    memory_store_bytes(dram, uram, (uint64_t) elements * memory_type_bytes(type));
}

static inline void memory_copy_device(void * destination, const void * source, uint64_t bytes) {
    if (memory_issue_linear_copy(reinterpret_cast<uint64_t>(destination), reinterpret_cast<uint64_t>(source), bytes)) {
        dayan_wait_lsu();
    }
}

static inline uint64_t memory_read_index(const void * indices, uint32_t index, uint32_t type, uint64_t scratch) {
    const uint32_t bytes = memory_type_bytes(type);
    if (type != kDayanDataTypeI32 && type != kDayanDataTypeI64) {
        DAYAN_PARAMS_ERROR();
    }
    configure_memory_lsu<1>(1, 0, 0, bytes, bytes);
    __tx_lsu_dma_d2u<1>((long) scratch, (long) ((const char *) indices + (uint64_t) index * bytes));
    dayan_wait_lsu();
    normalization_scalar_acquire((void *) (uintptr_t) scratch);
    if (type == kDayanDataTypeI32) {
        const int32_t value = *(const int32_t *) (uintptr_t) scratch;
        if (value < 0) {
            DAYAN_PARAMS_ERROR();
        }
        return (uint64_t) value;
    }
    if (type == kDayanDataTypeI64) {
        const int64_t value = *(const int64_t *) (uintptr_t) scratch;
        if (value < 0) {
            DAYAN_PARAMS_ERROR();
        }
        return (uint64_t) value;
    }
    DAYAN_PARAMS_ERROR();
    return 0;
}

static inline uint64_t memory_lowp_staging() {
    return (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kLowp;
}

template <void (*Convert)()>
static inline void memory_load_configured_lowp_to_f32(uint64_t output, const void * input, uint32_t elements) {
    const uint64_t staging = memory_lowp_staging();
    memory_load_bytes(staging, input, (uint64_t) elements * sizeof(uint16_t));
    ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(output, staging, elements);
}

static inline void memory_load_f32(uint64_t output, const void * input, uint32_t elements, uint32_t type) {
    if (type == kDayanDataTypeF32) {
        memory_load(output, input, elements, type);
        return;
    }
    if (type != kDayanDataTypeBF16 && type != kDayanDataTypeF16) {
        DAYAN_PARAMS_ERROR();
    }

    if (type == kDayanDataTypeBF16) {
        memory_load_configured_lowp_to_f32<__tx_vec_cvt_fp32_bf16<0, 0, 0>>(output, input, elements);
    } else {
        memory_load_configured_lowp_to_f32<__tx_vec_cvt_fp32_fp16<0, 0, 0>>(output, input, elements);
    }
}

static inline uint64_t memory_convert_from_f32(uint64_t output, uint64_t input, uint32_t elements, uint32_t type) {
    if (type == kDayanDataTypeF32) {
        if (output != input) {
            memory_copy_uram(output, input, elements * sizeof(float));
        }
        return output;
    }

    if (type == kDayanDataTypeBF16) {
        ggml_dayan_vec_detail::convert_f32_to_lowp<BaseNPUInstBuilder::DT_BF16, 1>(
            { input, (long) elements * (long) sizeof(float) },
            { input, (long) elements * (long) sizeof(float) },
            { output, (long) elements * (long) sizeof(uint16_t) }, 1, elements);
    } else if (type == kDayanDataTypeF16) {
        ggml_dayan_vec_detail::convert_f32_to_lowp<BaseNPUInstBuilder::DT_FP16, 1>(
            { input, (long) elements * (long) sizeof(float) },
            { input, (long) elements * (long) sizeof(float) },
            { output, (long) elements * (long) sizeof(uint16_t) }, 1, elements);
    } else {
        DAYAN_PARAMS_ERROR();
    }
    return output;
}

template <void (*Convert)()>
static inline void memory_copy_lowp_to_f32(const DayanCopyConfig * config,
                                           const void *            source,
                                           float *                 destination) {
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint32_t task        = dayan_native_task_id();
    const uint32_t tasks       = dayan_native_task_count();
    const uint64_t uram        = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t value_uram  = uram + NormalizationLayout::kValue;
    const uint64_t staging     = memory_lowp_staging();
    const uint64_t source_rows = config->elements / config->source_ne[0];
    const uint64_t tiles       = memory_tile_count(config->source_ne[0]);
    const uint64_t work_items  = source_rows * tiles;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint64_t row    = work / tiles;
        const uint32_t offset = (uint32_t) (work % tiles) * NormalizationLayout::kTileElements;
        uint64_t       index  = row;
        const uint32_t index1 = (uint32_t) (index % config->source_ne[1]);
        index /= config->source_ne[1];
        const uint32_t index2     = (uint32_t) (index % config->source_ne[2]);
        const uint32_t index3     = (uint32_t) (index / config->source_ne[2]);
        const char *   source_row = (const char *) source + (uint64_t) index1 * config->source_nb[1] +
                                  (uint64_t) index2 * config->source_nb[2] +
                                  (uint64_t) index3 * config->source_nb[3];
        const uint32_t elements = min_int((int) (config->source_ne[0] - offset),
                                          (int) NormalizationLayout::kTileElements);
        if (config->source_nb[0] == sizeof(uint16_t)) {
            memory_load_bytes(staging, source_row + (uint64_t) offset * sizeof(uint16_t),
                              (uint64_t) elements * sizeof(uint16_t));
        } else {
            configure_memory_lsu<1>(elements, config->source_nb[0], sizeof(uint16_t), sizeof(uint16_t),
                                    sizeof(uint16_t));
            __tx_lsu_dma_d2u<1>((long) staging,
                                (long) (source_row + (uint64_t) offset * config->source_nb[0]));
            dayan_wait_lsu();
        }
        ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(value_uram, staging, elements);
        normalization_store_f32(destination + row * config->source_ne[0] + offset, value_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void run_get_rows(const DayanRowsConfig * config,
                                const void *            source,
                                const void *            indices,
                                void *                  output) {
    memory_validate(config);
    if (source == nullptr || indices == nullptr || output == nullptr || config->row_elements == 0 ||
        config->rows == 0 ||
        (config->source_type != kDayanDataTypeF32 && config->source_type != kDayanDataTypeBF16 &&
         config->source_type != kDayanDataTypeF16) ||
        (config->destination_type != kDayanDataTypeF32 && config->destination_type != config->source_type)) {
        DAYAN_PARAMS_ERROR();
    }

    const uint64_t source_row_bytes = (uint64_t) config->row_elements * memory_type_bytes(config->source_type);
    const uint64_t output_row_bytes = (uint64_t) config->row_elements * memory_type_bytes(config->destination_type);
    const uint32_t task             = dayan_native_task_id();
    const uint32_t tasks            = dayan_native_task_count();
    const uint64_t source_address = (uint64_t) (uintptr_t) source;
    const uint64_t output_address = (uint64_t) (uintptr_t) output;
    const uint64_t source_bytes   = config->source_rows * source_row_bytes;
    const uint64_t output_bytes   = (uint64_t) config->rows * output_row_bytes;
    const uint64_t index_uram     = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kScalar;
    const bool disjoint = source_address < output_address ? output_address - source_address >= source_bytes :
                                                              source_address - output_address >= output_bytes;
    if (config->source_type == config->destination_type && disjoint) {
        bool issued = false;
        for (uint32_t row = task; row < config->rows; row += tasks) {
            uint32_t       linear   = row / config->index_ne[0];
            const uint32_t index1   = linear % config->index_ne[1];
            const uint32_t index2   = linear / config->index_ne[1];
            const uint64_t selected = memory_read_index(indices, row, config->index_type, index_uram);
            const uint64_t source_index =
                selected + (uint64_t) config->source_ne[0] * (index1 + (uint64_t) config->source_ne[1] * index2);
            if (selected >= config->source_ne[0] || source_index >= config->source_rows) {
                DAYAN_PARAMS_ERROR();
            }
            issued = memory_issue_linear_copy(output_address + (uint64_t) row * output_row_bytes,
                                              source_address + source_index * source_row_bytes, source_row_bytes) ||
                     issued;
        }
        if (issued) {
            dayan_wait_lsu();
        }
        return;
    }
    if (config->source_type == config->destination_type) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    if (config->source_type != kDayanDataTypeF32) {
        ggml_dayan_vec_detail::configure_f32_iterators();
    }
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint64_t tiles      = memory_tile_count(config->row_elements);
    const uint64_t work_items = (uint64_t) config->rows * tiles;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint32_t row      = (uint32_t) (work / tiles);
        const uint32_t offset   = (uint32_t) (work % tiles) * NormalizationLayout::kTileElements;
        const uint32_t elements = min_int((int) (config->row_elements - offset),
                                          (int) NormalizationLayout::kTileElements);
        uint32_t       linear   = row / config->index_ne[0];
        const uint32_t index1   = linear % config->index_ne[1];
        const uint32_t index2   = linear / config->index_ne[1];
        const uint64_t selected = memory_read_index(indices, row, config->index_type, index_uram);
        const uint64_t source_index =
            selected + (uint64_t) config->source_ne[0] * (index1 + (uint64_t) config->source_ne[1] * index2);
        if (selected >= config->source_ne[0] || source_index >= config->source_rows) {
            DAYAN_PARAMS_ERROR();
        }
        const char * source_row = (const char *) source + source_index * source_row_bytes;
        float *      output_row = (float *) output + (uint64_t) row * config->row_elements;
        if (config->source_type == kDayanDataTypeBF16) {
            memory_load_configured_lowp_to_f32<__tx_vec_cvt_fp32_bf16<0, 0, 0>>(
                out_uram, source_row + (uint64_t) offset * sizeof(uint16_t), elements);
        } else if (config->source_type == kDayanDataTypeF16) {
            memory_load_configured_lowp_to_f32<__tx_vec_cvt_fp32_fp16<0, 0, 0>>(
                out_uram, source_row + (uint64_t) offset * sizeof(uint16_t), elements);
        } else {
            memory_load(out_uram, source_row + (uint64_t) offset * sizeof(float), elements, config->source_type);
        }
        normalization_store_f32(output_row + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void run_set_rows(const DayanRowsConfig * config,
                                const float *           source,
                                const void *            indices,
                                void *                  destination) {
    memory_validate(config);
    if (source == nullptr || indices == nullptr || destination == nullptr || config->row_elements == 0 ||
        config->rows == 0 || config->source_type != kDayanDataTypeF32 ||
        (config->destination_type != kDayanDataTypeF32 && config->destination_type != kDayanDataTypeBF16 &&
         config->destination_type != kDayanDataTypeF16)) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = config->destination_type == kDayanDataTypeF32 ? uram + NormalizationLayout::kValue :
                                                                                uram + NormalizationLayout::kLowp;
    const uint64_t index_uram = uram + NormalizationLayout::kScalar;
    const uint64_t row_bytes  = (uint64_t) config->row_elements * memory_type_bytes(config->destination_type);
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    const uint64_t tiles      = memory_tile_count(config->row_elements);
    const uint64_t work_items = (uint64_t) config->rows * tiles;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint32_t row      = (uint32_t) (work / tiles);
        const uint32_t offset   = (uint32_t) (work % tiles) * NormalizationLayout::kTileElements;
        const uint32_t elements = min_int((int) (config->row_elements - offset),
                                          (int) NormalizationLayout::kTileElements);
        uint32_t       linear  = row;
        const uint32_t source1 = linear % config->source_ne[0];
        linear /= config->source_ne[0];
        const uint32_t source2  = linear % config->source_ne[1];
        const uint32_t source3  = linear / config->source_ne[1];
        const uint64_t index    = source1 + (uint64_t) config->index_ne[0] *
                                                ((source2 % config->index_ne[1]) +
                                                 (uint64_t) config->index_ne[1] * (source3 % config->index_ne[2]));
        const uint64_t selected = memory_read_index(indices, (uint32_t) index, config->index_type, index_uram);
        const uint64_t destination_index = selected + (uint64_t) config->destination_ne[0] *
                                                          (source2 + (uint64_t) config->destination_ne[1] * source3);
        if (selected >= config->destination_ne[0] || destination_index >= config->destination_rows) {
            DAYAN_PARAMS_ERROR();
        }
        const float * source_row      = source + (uint64_t) row * config->row_elements;
        char *        destination_row = (char *) destination + destination_index * row_bytes;
        normalization_load(input_uram, source_row + offset, elements);
        memory_convert_from_f32(out_uram, input_uram, elements, config->destination_type);
        memory_store(destination_row + (uint64_t) offset * memory_type_bytes(config->destination_type), out_uram,
                     elements, config->destination_type);
    }
}

static inline void run_copy(const DayanCopyConfig * config, const void * source, void * destination) {
    memory_validate(config);
    if (source == nullptr || destination == nullptr || config->elements == 0 ||
        (config->source_type != kDayanDataTypeF32 && config->source_type != kDayanDataTypeBF16 &&
         config->source_type != kDayanDataTypeF16) ||
        (config->destination_type != kDayanDataTypeF32 && config->destination_type != kDayanDataTypeBF16 &&
         config->destination_type != kDayanDataTypeF16)) {
        DAYAN_PARAMS_ERROR();
    }
    uint64_t source_elements = 1;
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->source_ne[dimension] == 0 || config->source_nb[dimension] == 0) {
            DAYAN_PARAMS_ERROR();
        }
        source_elements *= config->source_ne[dimension];
    }
    if (source_elements != config->elements) {
        DAYAN_PARAMS_ERROR();
    }

    const uint32_t source_bytes        = memory_type_bytes(config->source_type);
    const uint32_t destination_bytes   = memory_type_bytes(config->destination_type);
    const uint32_t task                = dayan_native_task_id();
    const uint32_t tasks               = dayan_native_task_count();
    const uint64_t total_bytes         = config->elements * source_bytes;
    const uint64_t source_address      = (uint64_t) (uintptr_t) source;
    const uint64_t destination_address = (uint64_t) (uintptr_t) destination;
    uint64_t source_span = source_bytes;
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        source_span += (uint64_t) (config->source_ne[dimension] - 1U) * config->source_nb[dimension];
    }
    const bool disjoint = source_address < destination_address ? destination_address - source_address >= source_span :
                                                                   source_address - destination_address >= total_bytes;
    if (config->source_type == config->destination_type && disjoint) {
        if (task != 0U) {
            return;
        }
        uint64_t destination_nb[4] = {
            destination_bytes,
            (uint64_t) config->source_ne[0] * destination_bytes,
            (uint64_t) config->source_ne[0] * config->source_ne[1] * destination_bytes,
            (uint64_t) config->source_ne[0] * config->source_ne[1] * config->source_ne[2] * destination_bytes,
        };
        const bool issued = memory_issue_tensor_copy(destination_address, source_address, config->source_ne,
                                                     config->source_nb, destination_nb, source_bytes);
        if (issued) {
            dayan_wait_lsu();
        }
        return;
    }

    if (config->destination_type == kDayanDataTypeF32 && config->source_type == kDayanDataTypeBF16) {
        memory_copy_lowp_to_f32<__tx_vec_cvt_fp32_bf16<0, 0, 0>>(config, source, (float *) destination);
        return;
    }
    if (config->destination_type == kDayanDataTypeF32 && config->source_type == kDayanDataTypeF16) {
        memory_copy_lowp_to_f32<__tx_vec_cvt_fp32_fp16<0, 0, 0>>(config, source, (float *) destination);
        return;
    }

    configure_normalization_lsu();
    if (config->source_type != kDayanDataTypeF32) {
        ggml_dayan_vec_detail::configure_f32_iterators();
    }
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t value_uram = uram + NormalizationLayout::kValue;
    const uint64_t out_uram =
        config->destination_type == kDayanDataTypeF32 ? value_uram : uram + NormalizationLayout::kLowp;
    const uint64_t source_rows = config->elements / config->source_ne[0];
    const uint64_t tiles       = memory_tile_count(config->source_ne[0]);
    const uint64_t work_items  = source_rows * tiles;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint64_t row    = work / tiles;
        const uint32_t offset = (uint32_t) (work % tiles) * NormalizationLayout::kTileElements;
        uint64_t       index  = row;
        const uint32_t index1 = (uint32_t) (index % config->source_ne[1]);
        index /= config->source_ne[1];
        const uint32_t index2     = (uint32_t) (index % config->source_ne[2]);
        const uint32_t index3     = (uint32_t) (index / config->source_ne[2]);
        const char *   source_row = (const char *) source + (uint64_t) index1 * config->source_nb[1] +
                                    (uint64_t) index2 * config->source_nb[2] + (uint64_t) index3 * config->source_nb[3];
        const uint64_t output_row = row * config->source_ne[0];
        const uint32_t elements = min_int((int) (config->source_ne[0] - offset),
                                          (int) NormalizationLayout::kTileElements);
        const uint64_t output_offset = output_row + offset;
        if (config->source_type == config->destination_type) {
            if (config->source_nb[0] == source_bytes) {
                memory_load(input_uram, source_row + (uint64_t) offset * source_bytes, elements,
                            config->source_type);
            } else {
                configure_memory_lsu<1>(elements, config->source_nb[0], source_bytes, source_bytes, source_bytes);
                __tx_lsu_dma_d2u<1>((long) input_uram,
                                    (long) (source_row + (uint64_t) offset * config->source_nb[0]));
                dayan_wait_lsu();
            }
            memory_store((char *) destination + output_offset * destination_bytes, input_uram, elements,
                         config->destination_type);
            continue;
        }
        if (config->source_nb[0] == source_bytes) {
            memory_load_f32(value_uram, source_row + (uint64_t) offset * source_bytes, elements,
                            config->source_type);
        } else {
            const uint64_t staging = config->source_type == kDayanDataTypeF32 ? value_uram :
                                                                                 memory_lowp_staging();
            configure_memory_lsu<1>(elements, config->source_nb[0], source_bytes, source_bytes, source_bytes);
            __tx_lsu_dma_d2u<1>((long) staging,
                                (long) (source_row + (uint64_t) offset * config->source_nb[0]));
            dayan_wait_lsu();
            if (config->source_type == kDayanDataTypeBF16) {
                ggml_dayan_vec_detail::issue_lowp_to_f32<__tx_vec_cvt_fp32_bf16<0, 0, 0>>(
                    value_uram, staging, elements);
            } else if (config->source_type == kDayanDataTypeF16) {
                ggml_dayan_vec_detail::issue_lowp_to_f32<__tx_vec_cvt_fp32_fp16<0, 0, 0>>(
                    value_uram, staging, elements);
            }
        }
        memory_convert_from_f32(out_uram, value_uram, elements, config->destination_type);
        memory_store((char *) destination + output_offset * destination_bytes, out_uram, elements,
                     config->destination_type);
    }
}

static inline void run_repeat(const DayanRepeatConfig * config, const void * source, void * output) {
    memory_validate(config);
    if (source == nullptr || output == nullptr ||
        (config->type != kDayanDataTypeF32 && config->type != kDayanDataTypeBF16 &&
         config->type != kDayanDataTypeF16)) {
        DAYAN_PARAMS_ERROR();
    }
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->source_ne[dimension] == 0 || config->output_ne[dimension] == 0 ||
            config->output_ne[dimension] % config->source_ne[dimension] != 0) {
            DAYAN_PARAMS_ERROR();
        }
    }

    if (dayan_native_task_id() != 0U) {
        return;
    }

    const uint64_t type_bytes = memory_type_bytes(config->type);
    uint64_t source_stride     = (uint64_t) config->source_ne[0] * type_bytes;
    uint64_t output_stride     = (uint64_t) config->output_ne[0] * type_bytes;
    MemoryCopyDimension dimensions[7] = {
        { config->output_ne[0] / config->source_ne[0], 0, source_stride },
        { config->source_ne[1], source_stride, output_stride },
        { config->output_ne[1] / config->source_ne[1], 0, (uint64_t) config->source_ne[1] * output_stride },
        {},
        {},
        {},
        {},
    };
    source_stride *= config->source_ne[1];
    output_stride *= config->output_ne[1];
    dimensions[3] = { config->source_ne[2], source_stride, output_stride };
    dimensions[4] = { config->output_ne[2] / config->source_ne[2], 0,
                      (uint64_t) config->source_ne[2] * output_stride };
    source_stride *= config->source_ne[2];
    output_stride *= config->output_ne[2];
    dimensions[5] = { config->source_ne[3], source_stride, output_stride };
    dimensions[6] = { config->output_ne[3] / config->source_ne[3], 0,
                      (uint64_t) config->source_ne[3] * output_stride };

    memory_issue_copy_dimensions(reinterpret_cast<uint64_t>(output), reinterpret_cast<uint64_t>(source),
                                 (uint64_t) config->source_ne[0] * type_bytes, dimensions, 7);
    dayan_wait_lsu();
}

static inline uint64_t memory_strided_offset(const uint64_t nb[4], const uint32_t indices[4]) {
    return (uint64_t) indices[0] * nb[0] + (uint64_t) indices[1] * nb[1] + (uint64_t) indices[2] * nb[2] +
           (uint64_t) indices[3] * nb[3];
}

static inline void run_concat(const DayanConcatConfig * config, const void * left, const void * right, void * output) {
    memory_validate(config);
    if (left == nullptr || right == nullptr || output == nullptr || config->axis >= 4 ||
        (config->type != kDayanDataTypeF32 && config->type != kDayanDataTypeBF16 &&
         config->type != kDayanDataTypeF16)) {
        DAYAN_PARAMS_ERROR();
    }
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->left_ne[dimension] == 0 || config->right_ne[dimension] == 0 || config->output_ne[dimension] == 0 ||
            (dimension == config->axis ?
                 config->output_ne[dimension] != config->left_ne[dimension] + config->right_ne[dimension] :
                 (config->output_ne[dimension] != config->left_ne[dimension] ||
                  config->output_ne[dimension] != config->right_ne[dimension]))) {
            DAYAN_PARAMS_ERROR();
        }
    }

    const uint32_t element_bytes = memory_type_bytes(config->type);
    const uint32_t task          = dayan_native_task_id();
    if (task != 0U) {
        return;
    }
    bool issued = memory_issue_tensor_copy(reinterpret_cast<uint64_t>(output), reinterpret_cast<uint64_t>(left),
                                           config->left_ne, config->left_nb, config->output_nb, element_bytes);
    const uint64_t right_offset = (uint64_t) config->left_ne[config->axis] * config->output_nb[config->axis];
    issued = memory_issue_tensor_copy(reinterpret_cast<uint64_t>(output) + right_offset,
                                      reinterpret_cast<uint64_t>(right), config->right_ne, config->right_nb,
                                      config->output_nb, element_bytes) ||
             issued;
    if (issued) {
        dayan_wait_lsu();
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif
