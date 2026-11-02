#pragma once

#include "ggml-dayan-normalization.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

template <long Group>
static inline void configure_memory_lsu(uint32_t transactions,
                                        uint64_t input_stride,
                                        uint64_t output_stride,
                                        uint32_t read_bytes,
                                        uint32_t write_bytes);

static constexpr float kDayanLog2E       = 1.4426950408889634f;
static constexpr float kDayanLn2         = 0.6931471805599453f;
static constexpr float kDayanGeluA       = 0.044715f;
static constexpr float kDayanGeluQuick   = 1.702f;
static constexpr float kDayanSqrt2OverPi = 0.7978845608028654f;

static inline void elementwise_validate(uint64_t elements, const void * input, void * output, const NativeContext & context = {}) {
    if (elements == 0 || input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    (void) dayan_native_task_id(context);
    DAYAN_PARAMS_IF(native_uram(context) == 0 || native_uram_size(context) < NormalizationLayout::kTotalBytes) {
        DAYAN_PARAMS_ERROR();
    }
}

using BinaryOperation = void (*)();

static inline uint64_t elementwise_tile_elements(uint64_t elements, uint32_t buffers, const NativeContext & context = {}) {
    constexpr uint64_t alignment = 512 / sizeof(float);
    const uint64_t capacity = native_uram_size(context) / (2 * buffers * sizeof(float)) / alignment * alignment;
    const uint64_t task_elements = align_up((elements + dayan_native_task_count(context) - 1) / dayan_native_task_count(context), alignment);
    return capacity < task_elements ? capacity : task_elements;
}

static inline bool binary_config_valid(const DayanBinaryConfig * config) {
    if (config->elements == 0 || config->ne[0] == 0) {
        return false;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < 4; ++dimension) {
        if (config->ne[dimension] == 0 || config->right_ne[dimension] == 0 ||
            config->ne[dimension] % config->right_ne[dimension] != 0) {
            return false;
        }
        elements *= config->ne[dimension];
    }
    if (elements != config->elements) {
        return false;
    }
    // A partial right row must divide one vector so the broadcast iterator can reuse complete lanes.
    if (config->right_ne[0] != config->ne[0] && (Constants::kVecLanes % config->right_ne[0]) != 0) {
        return false;
    }
    return true;
}

static inline bool binary_same_shape(const DayanBinaryConfig * config) {
    for (int dimension = 0; dimension < 4; ++dimension) {
        if (config->right_ne[dimension] != config->ne[dimension]) {
            return false;
        }
    }
    return true;
}

static inline bool binary_reusable_right(const DayanBinaryConfig * config) {
    return config->right_ne[0] <= Constants::kVecLanes && config->right_ne[1] == 1 &&
           config->right_ne[2] == 1 && config->right_ne[3] == 1;
}

static inline void binary_preload_right(const DayanBinaryConfig * config, const float * right, uint64_t right_uram) {
    const uint32_t row = config->right_ne[0];
    if (row == 1) {
        load_matrix<0>(sizeof(float), 1, sizeof(float), 1, right_uram, (uint64_t) (uintptr_t) right);
    } else if (row <= 32) {
        const uint32_t bytes  = row * sizeof(float);
        const uint32_t copies = Constants::kVecLanes / row;
        if (row <= 8) {
            for (uint32_t copy = 0; copy < copies; ++copy) {
                load_matrix<0>(bytes, 1, NormalizationLayout::kFloatBytes, 1, right_uram + copy * bytes,
                               (uint64_t) (uintptr_t) right);
            }
        } else if (row <= 16) {
            load_matrix<0>(bytes, 1, NormalizationLayout::kFloatBytes, 1, right_uram,
                           (uint64_t) (uintptr_t) right);
            for (uint32_t filled = 1; filled < copies; filled *= 2) {
                configure_memory_lsu<0>(filled, bytes, bytes, bytes, bytes);
                __tx_lsu_dma_u2u<0>((long) (right_uram + filled * bytes), (long) right_uram);
            }
        } else {
            load_matrix<0>(bytes, 1, NormalizationLayout::kFloatBytes, 1, right_uram,
                           (uint64_t) (uintptr_t) right);
            configure_memory_lsu<0>(2, 64, 64, 64, 64);
            __tx_lsu_dma_u2u<0>((long) (right_uram + bytes), (long) right_uram);
        }
    } else {
        load_matrix<0>((uint64_t) row * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1,
                       right_uram, (uint64_t) (uintptr_t) right);
    }
    dayan_wait_lsu();
}

template <BinaryOperation Operation>
static void run_binary_f32(const DayanBinaryConfig * config,
                           const float *             left,
                           const float *             right,
                           float *                   output, const NativeContext & context = {}) {
    DAYAN_PARAMS_IF(config == nullptr || right == nullptr || !binary_config_valid(config)) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, left, output, context);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = native_uram(context);
    const uint64_t left_uram  = uram + NormalizationLayout::kInput;
    const uint64_t right_uram = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id(context);
    const uint32_t tasks      = dayan_native_task_count(context);
    const bool repeated = binary_reusable_right(config) && !config->left_nb[0] &&
                          Constants::kVecLanes % config->right_ne[0] == 0;
    if (repeated || (binary_same_shape(config) && !config->left_nb[0] && !config->right_nb[0])) {
        const uint64_t tile = elementwise_tile_elements(config->elements, 3, context);
        const uint64_t bytes = tile * sizeof(float);
        const uint64_t first = (uint64_t) task * tile;
        if (first >= config->elements) {
            return;
        }
        if (repeated) {
            binary_preload_right(config, right, uram + bytes);
        }
        long checksum = 0;
        set_lsu_matrix<0>(checksum, 1, bytes, bytes, bytes);
        load_matrix<0>((config->elements - first) * sizeof(float), 1, bytes, 1, uram,
                       (uint64_t) (uintptr_t) (left + first));
        if (!repeated) {
            load_matrix<0>((config->elements - first) * sizeof(float), 1, bytes, 1, uram + bytes,
                           (uint64_t) (uintptr_t) (right + first));
        }
        dayan_wait_lsu();
        for (uint64_t offset = first, generation = 0; offset < config->elements;
             offset += (uint64_t) tasks * tile, ++generation) {
            const uint64_t base = uram + (generation & 1) * 3 * bytes;
            const uint64_t next_base = uram + ((generation + 1) & 1) * 3 * bytes;
            const uint64_t next = offset + (uint64_t) tasks * tile;
            const uint64_t elements = config->elements - offset < tile ? config->elements - offset : tile;
            if (next < config->elements) {
                load_matrix<0>((config->elements - next) * sizeof(float), 1, bytes, 1, next_base,
                               (uint64_t) (uintptr_t) (left + next));
                if (!repeated) {
                    load_matrix<0>((config->elements - next) * sizeof(float), 1, bytes, 1, next_base + bytes,
                                   (uint64_t) (uintptr_t) (right + next));
                }
            }
            ggml_dayan_vec_detail::configure_f32_tile(elements, repeated ? 0 : Constants::kVecBytes);
            constexpr long instructions = 4;
            ggml_dayan_vec_detail::begin_vec_block(instructions);
            __tx_vec_lsu_u2v<0, 0>((long) base);
            if (repeated && config->right_ne[0] == 1) {
                __tx_vec_lsu_u2v_broadcast32<1, 1>((long) (uram + bytes));
            } else {
                __tx_vec_lsu_u2v<1, 1>((long) ((repeated ? uram : base) + bytes));
            }
            Operation();
            __tx_vec_lsu_v2u<0, 3>((long) (base + 2 * bytes));
            ggml_dayan_vec_detail::start_vec_block(instructions);
            dayan_wait_vec();
            // Finish the next load and the previous store before reusing either slot.
            dayan_wait_lsu();
            store_matrix<0>(elements * sizeof(float), 1, bytes, 1, base + 2 * bytes,
                            (uint64_t) (uintptr_t) (output + offset));
        }
        dayan_wait_lsu();
        return;
    }


    if (native_dispatch(context, {config->elements, (config->elements + 8191) / 8192, 256U * 1024U}, [&](const NativeContext & worker) {
            run_binary_f32<Operation>(config, left, right, output, worker);
        })) { return; }
    if (!config->left_nb[0] && config->right_ne[0] == config->ne[0] &&
        config->right_ne[1] == 1 && config->right_ne[2] == 1 && config->right_ne[3] == 1 &&
        config->ne[0] % Constants::kVecLanes == 0 && config->ne[0] <= NormalizationLayout::kTileElements) {
        const uint64_t width = config->ne[0];
        const uint64_t rows = config->elements / width;
        const uint64_t task_rows = (rows + tasks - 1) / tasks;
        const uint64_t capacity_rows = 32768 / width;
        const uint64_t tile_rows = task_rows < capacity_rows ? task_rows : capacity_rows;
        const uint64_t tile_bytes = tile_rows * width * sizeof(float);
        const uint64_t weight = uram + tile_bytes;
        if ((uint64_t) task * tile_rows >= rows) { return; }
        long checksum = 0;
        set_lsu_matrix<0>(checksum, 1, tile_bytes, tile_bytes, tile_bytes);
        load_matrix<0>(width * sizeof(float), 1, tile_bytes, 1, weight, (uint64_t) (uintptr_t) right);
        dayan_wait_lsu();
        configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, width * sizeof(float));
        configure_vec_iterator<kVecV1Base>(0, 0, Constants::kVecBytes, 0);
        configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, width * sizeof(float));
        for (uint64_t row = (uint64_t) task * tile_rows; row < rows; row += (uint64_t) tasks * tile_rows) {
            const uint64_t count = rows - row < tile_rows ? rows - row : tile_rows;
            const uint64_t bytes = count * width * sizeof(float);
            load_matrix<0>(bytes, 1, tile_bytes, 1, uram, (uint64_t) (uintptr_t) (left + row * width));
            dayan_wait_lsu();
            configure_vec_grid(width / Constants::kVecLanes, count);
            constexpr long instructions = 4;
            begin_vec_block(instructions);
            __tx_vec_lsu_u2v<0, 0>(uram);
            __tx_vec_lsu_u2v<1, 1>(weight);
            Operation();
            __tx_vec_lsu_v2u<0, 3>(uram);
            start_vec_block(instructions);
            dayan_wait_vec();
            store_matrix<0>(bytes, 1, tile_bytes, 1, uram, (uint64_t) (uintptr_t) (output + row * width));
            dayan_wait_lsu();
        }
        return;
    }
    // Broadcast layout. Units never cross a row. The right operand is loaded
    // contiguously; the V1 iterator repeats it across the 64-lane grid steps:
    // a full right row advances by kVecBytes, a tiled right row (right row
    // width divides one vector register) re-reads the same lanes, and a scalar
    // right row uses the broadcast load.
    const uint64_t row_elements = config->ne[0];
    const uint64_t rows         = config->elements / row_elements;
    const uint64_t right_row    = config->right_ne[0];
    const bool     right_full   = right_row == row_elements;
    const bool     right_scalar = right_row == 1;
    const uint64_t unit         = row_elements < NormalizationLayout::kTileElements ? row_elements :
                                                                         NormalizationLayout::kTileElements;
    const uint64_t units_per_row = (row_elements + unit - 1) / unit;
    const uint64_t right_copies  = right_full || right_scalar ? 1 : Constants::kVecLanes / right_row;

    const bool reusable_right = binary_reusable_right(config);
    if (reusable_right && task < rows * units_per_row) {
        binary_preload_right(config, right, right_uram);
    }

    for (uint64_t unit_index = (uint64_t) task; unit_index < rows * units_per_row;
         unit_index += (uint64_t) tasks) {
        const uint64_t row     = unit_index / units_per_row;
        const uint64_t within  = unit_index % units_per_row;
        const uint64_t column  = within * unit;
        const uint64_t length  = row_elements - column < unit ? row_elements - column : unit;
        const uint64_t d1      = row % config->ne[1];
        const uint64_t d2      = (row / config->ne[1]) % config->ne[2];
        const uint64_t d3      = (row / ((uint64_t) config->ne[1] * config->ne[2])) % config->ne[3];
        const uint64_t r1      = d1 % config->right_ne[1];
        const uint64_t r2      = d2 % config->right_ne[2];
        const uint64_t r3      = d3 % config->right_ne[3];
        const uint64_t left_base = config->left_nb[0] ?
            (d1 * config->left_nb[0] + d2 * config->left_nb[1] + d3 * config->left_nb[2]) / sizeof(float) : row * row_elements;
        const uint64_t right_base = config->right_nb[0] ?
            (r1 * config->right_nb[0] + r2 * config->right_nb[1] + r3 * config->right_nb[2]) / sizeof(float) :
            r1 * right_row + r2 * right_row * config->right_ne[1] + r3 * right_row * config->right_ne[1] * config->right_ne[2];

        load_matrix<0>((uint64_t) length * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, left_uram,
                       (uint64_t) (uintptr_t) (left + left_base + column));
        if (!reusable_right) {
            if (right_full) {
                load_matrix<0>((uint64_t) length * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, right_uram,
                               (uint64_t) (uintptr_t) (right + right_base + column));
            } else {
                const uint64_t right_bytes =
                    right_scalar ? sizeof(float) : right_row * sizeof(float);
                for (uint64_t copy = 0; copy < right_copies; ++copy) {
                    load_matrix<0>(right_bytes, 1, NormalizationLayout::kFloatBytes, 1,
                                   right_uram + copy * right_bytes, (uint64_t) (uintptr_t) (right + right_base));
                }
            }
        }
        dayan_wait_lsu();
        ggml_dayan_vec_detail::configure_f32_tile(length, right_full ? (long) Constants::kVecBytes : 0);
        constexpr long instructions = 4;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) left_uram);
        if (right_scalar) {
            __tx_vec_lsu_u2v_broadcast32<1, 1>((long) right_uram);
        } else {
            __tx_vec_lsu_u2v<1, 1>((long) right_uram);
        }
        Operation();
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        store_matrix<0>((uint64_t) length * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, out_uram,
                        (uint64_t) (uintptr_t) (output + row * row_elements + column));
        dayan_wait_lsu();
    }
}

static void run_div_f32(const DayanBinaryConfig * config,
                        const float *             left,
                        const float *             right,
                        float *                   output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {config->elements, (config->elements + 8191) / 8192, 256U * 1024U}, [&](const NativeContext & worker) {
            run_div_f32(config, left, right, output, worker);
        })) { return; }

    DAYAN_PARAMS_IF(config == nullptr || right == nullptr || !binary_config_valid(config)) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, left, output, context);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = native_uram(context);
    const uint64_t left_uram  = uram + NormalizationLayout::kInput;
    const uint64_t right_uram = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id(context);
    const uint32_t tasks      = dayan_native_task_count(context);
    ggml_dayan_vec_detail::configure_f32_tile(NormalizationLayout::kTileElements);

    if (binary_same_shape(config) && !config->left_nb[0] && !config->right_nb[0]) {
        for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
             offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
            const uint64_t remaining = config->elements - offset;
            const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                           (uint32_t) remaining :
                                           (uint32_t) NormalizationLayout::kTileElements;
            load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, left_uram,
                           (uint64_t) (uintptr_t) (left + offset));
            load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, right_uram,
                           (uint64_t) (uintptr_t) (right + offset));
            dayan_wait_lsu();
            constexpr long instructions = 5;
            ggml_dayan_vec_detail::begin_vec_block(instructions);
            __tx_vec_lsu_u2v<0, 0>((long) left_uram);
            __tx_vec_lsu_u2v<1, 1>((long) right_uram);
            __tx_vec_v_fp32_sfu<1, 16, 5>();
            __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 3>((long) out_uram);
            ggml_dayan_vec_detail::start_vec_block(instructions);
            dayan_wait_vec();
            store_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, out_uram,
                            (uint64_t) (uintptr_t) (output + offset));
            dayan_wait_lsu();
        }
        return;
    }

    // Broadcast layout, mirroring run_binary_f32: the reciprocal is taken on
    // the right operand after a scalar, tiled, or full-row load.
    const uint64_t row_elements = config->ne[0];
    const uint64_t rows         = config->elements / row_elements;
    const uint64_t right_row    = config->right_ne[0];
    const bool     right_full   = right_row == row_elements;
    const bool     right_scalar = right_row == 1;
    const uint64_t unit         = row_elements < NormalizationLayout::kTileElements ? row_elements :
                                                                         NormalizationLayout::kTileElements;
    const uint64_t units_per_row = (row_elements + unit - 1) / unit;
    const uint64_t right_copies  = right_full || right_scalar ? 1 : Constants::kVecLanes / right_row;
    const bool reusable_right = binary_reusable_right(config);
    const bool packed_scalars = right_scalar && !config->right_nb[0] &&
                                config->right_ne[1] == config->ne[1] &&
                                config->right_ne[2] == config->ne[2] &&
                                config->right_ne[3] == config->ne[3] && !reusable_right;
    uint64_t scalar_first = rows;
    uint64_t scalar_count = 0;
    if (reusable_right && task < rows * units_per_row) {
        binary_preload_right(config, right, right_uram);
    }

    for (uint64_t unit_index = (uint64_t) task; unit_index < rows * units_per_row;
         unit_index += (uint64_t) tasks) {
        const uint64_t row     = unit_index / units_per_row;
        const uint64_t within  = unit_index % units_per_row;
        const uint64_t column  = within * unit;
        const uint64_t length  = row_elements - column < unit ? row_elements - column : unit;
        const uint64_t d1      = row % config->ne[1];
        const uint64_t d2      = (row / config->ne[1]) % config->ne[2];
        const uint64_t d3      = (row / ((uint64_t) config->ne[1] * config->ne[2])) % config->ne[3];
        const uint64_t r1      = d1 % config->right_ne[1];
        const uint64_t r2      = d2 % config->right_ne[2];
        const uint64_t r3      = d3 % config->right_ne[3];
        const uint64_t left_base = config->left_nb[0] ?
            (d1 * config->left_nb[0] + d2 * config->left_nb[1] + d3 * config->left_nb[2]) / sizeof(float) : row * row_elements;
        const uint64_t right_base = config->right_nb[0] ?
            (r1 * config->right_nb[0] + r2 * config->right_nb[1] + r3 * config->right_nb[2]) / sizeof(float) :
            r1 * right_row + r2 * right_row * config->right_ne[1] + r3 * right_row * config->right_ne[1] * config->right_ne[2];

        load_matrix<0>((uint64_t) length * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, left_uram,
                       (uint64_t) (uintptr_t) (left + left_base + column));
        uint64_t scalar_uram = right_uram;
        if (packed_scalars) {
            if (row < scalar_first || row >= scalar_first + scalar_count) {
                scalar_first = row;
                scalar_count = rows - row < NormalizationLayout::kTileElements ? rows - row : NormalizationLayout::kTileElements;
                // Keep consecutive per-token divisors in URAM across row iterations.
                load_matrix<0>(scalar_count * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1,
                               right_uram, (uint64_t) (uintptr_t) (right + row));
            }
            scalar_uram += (row - scalar_first) * sizeof(float);
        } else if (!reusable_right) {
            if (right_full) {
                load_matrix<0>((uint64_t) length * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, right_uram,
                               (uint64_t) (uintptr_t) (right + right_base + column));
            } else {
                const uint64_t right_bytes =
                    right_scalar ? sizeof(float) : right_row * sizeof(float);
                for (uint64_t copy = 0; copy < right_copies; ++copy) {
                    load_matrix<0>(right_bytes, 1, NormalizationLayout::kFloatBytes, 1,
                                   right_uram + copy * right_bytes, (uint64_t) (uintptr_t) (right + right_base));
                }
            }
        }
        dayan_wait_lsu();
        ggml_dayan_vec_detail::configure_f32_tile(length, right_full ? (long) Constants::kVecBytes : 0);
        constexpr long instructions = 5;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) left_uram);
        if (packed_scalars) {
            // Read packed words through the scalar port before broadcasting.
            __tx_vec_lsu_g2v_broadcast32<1, 1>((long) *(const volatile uint32_t *) (uintptr_t) scalar_uram);
        } else if (right_scalar) {
            __tx_vec_lsu_u2v_broadcast32<1, 1>((long) right_uram);
        } else {
            __tx_vec_lsu_u2v<1, 1>((long) right_uram);
        }
        __tx_vec_v_fp32_sfu<1, 16, 5>();
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        store_matrix<0>((uint64_t) length * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, out_uram,
                        (uint64_t) (uintptr_t) (output + row * row_elements + column));
        dayan_wait_lsu();
    }
}

static inline void sigmoid_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 8;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void silu_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 10;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void softplus_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 20;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLn2));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(0.0f));
    __tx_vec_vv_fp32_cmp<16, 17, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 17, 0>();
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void square_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 3;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void sqrt_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 4>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void log_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 5;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLn2));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void sin_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 3;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 2>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void cos_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 3;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 3>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void abs_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 7;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void neg_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void relu_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 5;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(0.0f));
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void exp_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 5;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void tanh_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 12;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-2.0f * kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(2.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void gelu_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 18;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanGeluA));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>(
        (long) ggml_dayan_vec_detail::f32_bits(-2.0f * kDayanLog2E * kDayanSqrt2OverPi));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void gelu_quick_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 10;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>(
        (long) ggml_dayan_vec_detail::f32_bits(-kDayanGeluQuick * kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

using UnaryOperation = void (*)(uint64_t, uint64_t);

template <UnaryOperation Operation>
static void run_unary_f32(const DayanElementwiseConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, input, output, context);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = native_uram(context);
    const uint32_t task       = dayan_native_task_id(context);
    const uint32_t tasks      = dayan_native_task_count(context);
    const uint64_t tile = elementwise_tile_elements(config->elements, 2, context);
    const uint64_t bytes = tile * sizeof(float);
    const uint64_t first = (uint64_t) task * tile;
    if (first >= config->elements) {
        return;
    }
    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, bytes, bytes, bytes);
    load_matrix<0>((config->elements - first) * sizeof(float), 1, bytes, 1, uram,
                   (uint64_t) (uintptr_t) (input + first));
    dayan_wait_lsu();
    for (uint64_t offset = first, generation = 0; offset < config->elements;
         offset += (uint64_t) tasks * tile, ++generation) {
        const uint64_t base = uram + (generation & 1) * 2 * bytes;
        const uint64_t next_base = uram + ((generation + 1) & 1) * 2 * bytes;
        const uint64_t next = offset + (uint64_t) tasks * tile;
        const uint64_t elements = config->elements - offset < tile ? config->elements - offset : tile;
        if (next < config->elements) {
            load_matrix<0>((config->elements - next) * sizeof(float), 1, bytes, 1, next_base,
                           (uint64_t) (uintptr_t) (input + next));
        }
        ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
        Operation(base + bytes, base);
        dayan_wait_lsu();
        store_matrix<0>(elements * sizeof(float), 1, bytes, 1, base + bytes,
                        (uint64_t) (uintptr_t) (output + offset));
    }
    dayan_wait_lsu();
}

static inline void run_reduce_rows_f32(const DayanReduceConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {(uint64_t) config->width * config->rows, config->rows, 256U * 1024U}, [&](const NativeContext & worker) {
            run_reduce_rows_f32(config, input, output, worker);
        })) { return; }

    if (config == nullptr || config->width == 0 || config->rows == 0 || input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    const uint32_t task  = dayan_native_task_id(context);
    const uint32_t tasks = dayan_native_task_count(context);
    DAYAN_PARAMS_IF(native_uram(context) == 0 || native_uram_size(context) < NormalizationLayout::kTotalBytes) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    const uint64_t uram        = native_uram(context);
    const uint64_t input_uram  = uram + NormalizationLayout::kInput;
    const uint64_t accumulator = uram + NormalizationLayout::kVectorScalar;
    for (uint32_t row = task; row < config->rows; row += tasks) {
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input + (uint64_t) row * config->width + offset, elements, context);
            normalization_sum_accumulate(accumulator, input_uram, elements, offset == 0);
        }
        constexpr long instructions = 4;
        ggml_dayan_vec_detail::configure_vec_grid(1);
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v_broadcast32<0, 0>((long) accumulator);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->scale_bits);
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 3>((long) accumulator);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        normalization_store_f32(output + row, accumulator, 1);
        dayan_wait_lsu();
    }
}

static inline void run_scale_f32(const DayanScaleConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {config->elements, (config->elements + 8191) / 8192, 256U * 1024U}, [&](const NativeContext & worker) {
            run_scale_f32(config, input, output, worker);
        })) { return; }

    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    // zero-element scale is the decode no-op that keeps the mega parameter
    // layout aligned: nothing to read or write
    if (config->elements == 0) {
        return;
    }
    elementwise_validate(config->elements, input, output, context);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = native_uram(context);
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id(context);
    const uint32_t tasks      = dayan_native_task_count(context);
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        normalization_load(input_uram, input + offset, elements, context);
        ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
        const bool has_bias     = (config->bias_bits & 0x7fffffffU) != 0U;
        const long instructions = has_bias ? 6 : 4;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) input_uram);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->scale_bits);
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        if (has_bias) {
            __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->bias_bits);
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
        }
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void run_clamp_f32(const DayanClampConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {config->elements, (config->elements + 8191) / 8192, 256U * 1024U}, [&](const NativeContext & worker) {
            run_clamp_f32(config, input, output, worker);
        })) { return; }

    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, input, output, context);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = native_uram(context);
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id(context);
    const uint32_t tasks      = dayan_native_task_count(context);
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        normalization_load(input_uram, input + offset, elements, context);
        ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
        constexpr long instructions = 8;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) input_uram);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->min_bits);
        __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->max_bits);
        __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
        __tx_vec_vv_mux2<0, 16, 0, 0>();
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void l2_norm_fused(uint64_t input, uint64_t output, uint32_t elements, uint32_t epsilon_bits) {
    const long     chunks = ggml_dayan_vec_detail::f32_chunks(elements);
    const uint32_t tail   = elements % Constants::kVecLanes;
    const long     instructions = 6 + 18 * chunks + (tail == 0 ? 0 : 2);

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    for (long chunk = 0; chunk < chunks; ++chunk) {
        const uint64_t chunk_input = input + (uint64_t) chunk * Constants::kVecBytes;
        if (tail != 0 && chunk + 1 == chunks) {
            __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
            __tx_vec_lsu_g2s<0>((long) ((uint64_t{ 1 } << tail) - 1U));
            __tx_vec_lsu_u2v_mask<0, 0, 0>((long) chunk_input);
        } else {
            __tx_vec_lsu_u2v<0, 0>((long) chunk_input);
        }
        __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
        ggml_dayan_vec_detail::emit_f32_sum_stages();
        __tx_vec_vv_fp32_add<1, 0, 22, 0>();
    }
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) epsilon_bits);
    __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<1, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<1, 16, 4>();
    for (long chunk = 0; chunk < chunks; ++chunk) {
        const uint64_t offset = (uint64_t) chunk * Constants::kVecBytes;
        __tx_vec_lsu_u2v<0, 0>((long) (input + offset));
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 0>((long) (output + offset));
    }
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void run_l2_norm_generic(const DayanNormConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    const uint64_t uram       = native_uram(context);
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint64_t scalar     = uram + NormalizationLayout::kVectorScalar;

    for (uint32_t row = dayan_native_task_id(context); row < config->rows; row += dayan_native_task_count(context)) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        const float *  input_row  = (const float *) normalization_input_row(config, input, row);
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements, context);
            normalization_square_sum_accumulate(scalar, input_uram, elements, offset == 0);
        }

        constexpr long finalize_instructions = 7;
        ggml_dayan_vec_detail::configure_vec_grid(1);
        ggml_dayan_vec_detail::begin_vec_block(finalize_instructions);
        __tx_vec_lsu_u2v_broadcast32<0, 0>((long) scalar);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->epsilon_bits);
        __tx_vec_vv_fp32_mul<1, 16, 16, 0>();
        __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_v_fp32_sfu<0, 0, 4>();
        __tx_vec_lsu_v2u<0, 3>((long) scalar);
        ggml_dayan_vec_detail::start_vec_block(finalize_instructions);
        dayan_wait_vec();

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements, context);
            normalization_apply_rms(out_uram, input_uram, scalar, elements);
            normalization_store_f32(output + row_offset + offset, out_uram, elements);
            dayan_wait_lsu();
        }
    }
}

static inline void l2_norm_load_rows(const DayanNormConfig * config, const float * input,
                                    uint64_t destination, uint32_t first, uint32_t rows) {
    const uint64_t row_bytes = (uint64_t) config->width * sizeof(float);
    long checksum = 0;
    for (uint32_t row = 0; row < rows;) {
        const uint32_t index = first + row;
        const uint32_t head = index % config->input_ne[1];
        const uint32_t token = index / config->input_ne[1] % config->input_ne[2];
        uint32_t count;
        if (head == 0 && rows - row >= config->input_ne[1] && config->input_nb[1] == row_bytes) {
            const uint32_t groups = min_int((rows - row) / config->input_ne[1], config->input_ne[2] - token);
            count = groups * config->input_ne[1];
            const uint64_t group_bytes = config->input_ne[1] * row_bytes;
            set_lsu_matrix<0>(checksum, groups, group_bytes, config->input_nb[2], group_bytes);
        } else {
            count = min_int(rows - row, config->input_ne[1] - head);
            set_lsu_matrix<0>(checksum, count, row_bytes, config->input_nb[1], row_bytes);
        }
        __tx_lsu_dma_d2u<0>((long) (destination + row * row_bytes),
                           (long) normalization_input_row(config, input, index));
        row += count;
    }
}

static inline void run_l2_norm(const DayanNormConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (config->width > 4U * Constants::kVecLanes || config->width % Constants::kVecLanes != 0) {
        if (native_dispatch(context, {(uint64_t) config->width * config->rows, config->rows, 256U * 1024U}, [&](const NativeContext & worker) {
                run_l2_norm(config, input, output, worker);
            })) { return; }
    }

    normalization_validate<kDayanDataTypeF32>(config, context);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    if (config->width > 4U * Constants::kVecLanes) {
        run_l2_norm_generic(config, input, output, context);
        return;
    }

    const uint64_t uram        = native_uram(context);
    const uint64_t input_uram  = uram + NormalizationLayout::kInput;
    if ((config->width % Constants::kVecLanes) == 0U) {
        const uint32_t row_bytes     = config->width * sizeof(float);
        const uint32_t task  = dayan_native_task_id(context);
        const uint32_t tasks = dayan_native_task_count(context);
        const uint32_t rows_per_tile = min_int(native_uram_size(context) / (4U * row_bytes),
                                               ((uint64_t) config->rows + tasks - 1) / tasks);
        const uint64_t bytes = (uint64_t) rows_per_tile * row_bytes;
        const uint64_t first = (uint64_t) task * rows_per_tile;
        if (first >= config->rows) {
            return;
        }
        configure_vec_iterator<kVecV0Base>(0, 0, 0, row_bytes);
        configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
        configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
        configure_vec_iterator<kVecV3Base>(0, 0, 0, row_bytes);

        l2_norm_load_rows(config, input, uram, first, min_int(config->rows - first, rows_per_tile));
        dayan_wait_lsu();
        for (uint64_t row_base = first, generation = 0; row_base < config->rows;
             row_base += (uint64_t) tasks * rows_per_tile, ++generation) {
            const uint32_t rows = min_int((int) (config->rows - row_base), (int) rows_per_tile);
            const uint64_t base = uram + (generation & 1) * 2 * bytes;
            const uint64_t next = row_base + (uint64_t) tasks * rows_per_tile;
            if (next < config->rows) {
                l2_norm_load_rows(config, input, uram + ((generation + 1) & 1) * 2 * bytes,
                                  next, min_int(config->rows - next, rows_per_tile));
            }
            ggml_dayan_vec_detail::configure_vec_grid(1, rows);
            l2_norm_fused(base, base + bytes, config->width, config->epsilon_bits);
            dayan_wait_lsu();
            long checksum = 0;
            set_lsu_matrix<0>(checksum, 1, (uint64_t) rows * row_bytes, 0, 0);
            __tx_lsu_dma_u2d<0>((long) (output + row_base * config->width), (long) (base + bytes));
        }
        dayan_wait_lsu();
        return;
    }

    const uint64_t output_uram = uram + NormalizationLayout::kValue;
    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    for (uint32_t row = dayan_native_task_id(context); row < config->rows; row += dayan_native_task_count(context)) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        const float *  input_row  = (const float *) normalization_input_row(config, input, row);
        load_matrix<0>((uint64_t) config->width * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, input_uram,
                       (uint64_t) (uintptr_t) input_row);
        dayan_wait_lsu();
        l2_norm_fused(input_uram, output_uram, config->width, config->epsilon_bits);
        normalization_store_f32(output + row_offset, output_uram, config->width);
        dayan_wait_lsu();
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif
