#pragma once

#include "ggml-dayan-mxfp4.h"
#include "ggml-dayan-normalization.h"

struct DayanFfnLayout {
    enum : int { rows = 8, columns = 256, c_stride = columns * 4 };
    uint64_t a, as, hidden, hidden_scales, b[2], bs[2], c[2], sum, bytes;
    int inner, b_stride, a_stride, hidden_stride;

    static DayanFfnLayout make(const DayanFfnConfig & config) {
        const bool mx = config.gate.weight_type == kDayanDataTypeMXFP4;
        DayanFfnLayout layout = {};
        layout.inner = mx ? 1024 : 256;
        layout.b_stride = mx ? layout.inner / 2 : layout.inner * 2;
        layout.a_stride = mx ? config.gate.k / 2 : config.gate.k * 2;
        layout.hidden_stride = mx ? config.gate.n / 2 : config.gate.n * 2;
        uint64_t address = 0;
        layout.a = address; address += rows * layout.a_stride;
        layout.as = address; address += mx ? config.gate.k / 128 * 64 : 0;
        address = (address + 1023) & ~1023ULL;
        layout.hidden = address; address += rows * layout.hidden_stride;
        layout.hidden_scales = address; address += mx ? config.gate.n / 128 * 64 : 0;
        for (int i = 0; i < 2; ++i) {
            address = (address + 1023) & ~1023ULL;
            layout.b[i] = address; address += 2 * columns * layout.b_stride;
            layout.bs[i] = address; address += mx ? 2 * columns * layout.inner / 32 : 0;
        }
        address = (address + 1023) & ~1023ULL;
        for (int i = 0; i < 2; ++i) {
            layout.c[i] = address; address += rows * c_stride;
        }
        layout.sum = address; address += config.down.n * 4;
        layout.bytes = (address + 1023) & ~1023ULL;
        return layout;
    }
};

#if defined(__DAYAN_DEVICE__)
namespace ggml_dayan_moe_ffn_detail {

using namespace ggml_dayan_mxfp4_detail;

static constexpr uint64_t kWeightSwizzle = MAT_BATCH_SWIZZLE(6, 3, 3);

struct State {
    const DayanFfnConfig * config;
    DayanFfnLayout layout;
    uint64_t weight[3], scales[3];
    int32_t ids[32];
    uint64_t physical, local;
};

static void load_weights(void * argument) {
    State & s = *(State *) argument;
    const auto & l = s.layout;
    const bool mx = s.config->gate.weight_type == kDayanDataTypeMXFP4;
    uint64_t sequence = 0;
    long checksum = 0;
    for (uint32_t expert = 0; expert < s.config->experts_used; ++expert) {
        for (int stage = 0; stage < 2; ++stage) {
            const auto & cfg = stage ? s.config->down : s.config->gate;
            for (uint64_t col = 0; col < cfg.n; col += l.columns) {
                const int columns = extent(cfg.n - col, l.columns);
                for (uint64_t step = 0; step < cfg.k; step += l.inner) {
                    const int slot = sequence++ & 1;
                    const int inner = extent(cfg.k - step, l.inner);
                    const auto resource = DAYAN_LOCAL_RESOURCE(slot);
                    matmul_check(dayan_resource_wait_free(resource));
                    for (int product = 0; product < (stage ? 1 : 2); ++product) {
                        const int operand = stage ? 2 : product;
                        const uint64_t row = (uint64_t) s.ids[expert] * cfg.n + col;
                        const uint64_t target = s.local + l.b[slot] + product * l.columns * l.b_stride;
                        set_lsu_matrix<0>(checksum, columns, l.b_stride, cfg.b_stride, l.b_stride);
                        __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(kWeightSwizzle);
                        load_matrix<0>(mx ? inner / 2 : inner * 2, columns, l.b_stride, columns, target,
                                       s.weight[operand] + row * cfg.b_stride + (mx ? step / 2 : step * 2));
                        if (mx) {
                            const uint64_t scales = s.local + l.bs[slot] + product * l.columns * l.inner / 32;
                            set_lsu_matrix<0>(checksum, inner / 128, columns * 4, cfg.b_scale_stride, l.columns * 4);
                            load_matrix<0>(columns * 4, inner / 128, columns * 4, inner / 128, scales,
                                           s.scales[operand] + step / 128 * cfg.b_scale_stride + row * 4);
                        }
                    }
                    matmul_check(dayan_resource_produce_lsu(resource, 1, 0));
                }
            }
        }
    }
    dayan_wait_lsu();
}

static int run(const DayanFfnConfig * config, const void * input, const void * input_scales,
               const void * gate, const void * gate_scales, const void * up, const void * up_scales,
               const void * down, const void * down_scales, const int32_t * selected, const float * weights, void * output) {
    if (!config || !input || !gate || !up || !down || !output || config->gate.m != 1 || config->down.m != 1 ||
        !config->gate.k || !config->gate.n || !config->down.n || config->gate.k > 65536 || config->gate.n > 65536 || config->down.n > 65536 ||
        config->gate.k % 128 || config->gate.n % 128 || config->down.n % 64 ||
        config->down.k != config->gate.n || config->gate.weight_type != config->down.weight_type ||
        (config->gate.weight_type != kDayanDataTypeMXFP4 && config->gate.weight_type != kDayanDataTypeBF16) ||
        !config->experts || !config->experts_used || config->experts_used > 32 || config->experts_used > config->experts ||
        (config->experts > 1 && (!selected || !weights))) { return -22; }
    State s = {};
    s.config = config;
    s.layout = DayanFfnLayout::make(*config);
    s.physical = (uintptr_t) dayan_uram_base();
    s.local = s.physical & (Constants::kUramWindowStride - 1);
    const auto & l = s.layout;
    const bool mx = config->gate.weight_type == kDayanDataTypeMXFP4;
    if (l.bytes > dayan_uram_size() || !s.physical || s.local + l.bytes > Constants::kUramWindowStride ||
        (mx && (!input_scales || !gate_scales || !up_scales || !down_scales)) ||
        (config->down.output_type != kDayanDataTypeF32 && config->down.output_type != kDayanDataTypeBF16)) { return -12; }
    s.weight[0] = (uintptr_t) gate; s.weight[1] = (uintptr_t) up; s.weight[2] = (uintptr_t) down;
    s.scales[0] = (uintptr_t) gate_scales; s.scales[1] = (uintptr_t) up_scales; s.scales[2] = (uintptr_t) down_scales;
    if (config->experts > 1) {
        dayan_cache_invalidate(selected, config->experts_used * sizeof(int32_t));
        dayan_cache_invalidate(weights, config->experts_used * sizeof(float));
    }
    for (uint32_t e = 0; e < config->experts_used; ++e) {
        s.ids[e] = config->experts > 1 ? selected[e] : 0;
        if (s.ids[e] < 0 || (uint32_t) s.ids[e] >= config->experts) { return -22; }
    }
    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, l.a_stride, config->gate.a_stride, l.a_stride);
    load_matrix<0>(l.a_stride, 1, l.a_stride, 1, s.local + l.a, (uintptr_t) input);
    if (mx) {
        set_lsu_matrix<0>(checksum, config->gate.k / 128, 64, config->gate.a_scale_stride, 64);
        load_matrix<0>(32, config->gate.k / 128, 64, config->gate.k / 128, s.local + l.as, (uintptr_t) input_scales);
    }
    dayan_wait_lsu();
    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = 1;
    resources.local_resource_count = 2;
    matmul_check(dayan_create_thread("ffn-load", load_weights, &s, &resources));
    uint64_t sequence = 0, phases = 0;
    for (uint32_t expert = 0; expert < config->experts_used; ++expert) {
        for (int stage = 0; stage < 2; ++stage) {
            const auto & cfg = stage ? config->down : config->gate;
            __tx_mat_tcsr_write<MAT_ASTRIDE>(stage ? l.hidden_stride : l.a_stride);
            __tx_mat_tcsr_write<MAT_BSTRIDE>(l.b_stride);
            __tx_mat_tcsr_write<MAT_CSTRIDE>(l.c_stride);
            __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(0);
            __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(kWeightSwizzle);
            __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(0);
            __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
            __tx_mat_tcsr_write<MAT_ASCALESTRIDE>(64);
            __tx_mat_tcsr_write<MAT_BSCALESTRIDE>(l.columns * 4);
            __tx_mat_tcsr_write<MAT_DATATYPE>(mx ? MAT_DATATYPE_PAIR(0x38, 0x38) : BaseNPUInstBuilder::DT_BF16);
            __tx_mat_tcsr_write<MAT_M>(0);
            for (uint64_t col = 0; col < cfg.n; col += l.columns) {
                const int columns = extent(cfg.n - col, l.columns);
                __tx_mat_tcsr_write<MAT_N>(columns - 1);
                for (uint64_t step = 0; step < cfg.k; step += l.inner) {
                    const int slot = sequence++ & 1;
                    const auto mask = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(slot));
                    matmul_check(dayan_resource_wait(0, mask, phases, 0));
                    __tx_mat_tcsr_write<MAT_K>(extent(cfg.k - step, l.inner) - 1);
                    const uint64_t a = s.local + (stage ? l.hidden : l.a) + (mx ? step / 2 : step * 2);
                    const uint64_t as = s.local + (stage ? l.hidden_scales : l.as) + step / 128 * 64;
                    for (int product = 0; product < (stage ? 1 : 2); ++product) {
                        const uint64_t b = s.local + l.b[slot] + product * l.columns * l.b_stride;
                        const uint64_t c = s.local + l.c[product];
                        __tx_mat_setptr<0>(0, as, s.local + l.bs[slot] + product * l.columns * l.inner / 32);
                        if (!step) { __tx_mat_mm<0>(c, a, b); }
                        else { __tx_mat_mma<0>(c, a, b); }
                    }
                    matmul_check(dayan_resource_consume_mat(mask));
                    phases ^= mask;
                }
                dayan_wait_mat();
                const uint64_t c = s.physical + l.c[0];
                if (!stage) {
                    post_values<2>(c, s.physical + l.c[1], 1, columns, l.c_stride,
                                   mx ? kDayanDataTypeMXFP4 : kDayanDataTypeBF16);
                    dayan_wait_vec();
                    if (mx) {
                        pack_quantized(c, s.physical + l.hidden + col / 2, s.physical + l.hidden_scales + col / 128 * 64,
                                       1, columns, l.c_stride, 128, l.hidden_stride, 64, 0);
                    } else {
                        set_lsu_matrix<0>(checksum, columns / 32, 64, 128, 64);
                        __tx_lsu_dma_u2u<0>(s.physical + l.hidden + col * 2, c);
                    }
                    dayan_wait_lsu();
                } else {
                    using namespace ggml_dayan_vec_detail;
                    const uint64_t accumulator = s.physical + l.sum + col * 4;
                    configure_f32_iterators();
                    configure_vec_grid(columns / Constants::kVecLanes);
                    const long instructions = expert ? 6 : 4;
                    begin_vec_block(instructions);
                    __tx_vec_lsu_u2v<0, 0>(c);
                    __tx_vec_lsu_g2v_broadcast32<1, 2>(f32_bits(config->experts > 1 ? weights[expert] : 1.0f));
                    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
                    if (expert) {
                        __tx_vec_lsu_u2v<1, 1>(accumulator);
                        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
                    }
                    __tx_vec_lsu_v2u<0, 3>(accumulator);
                    start_vec_block(instructions);
                    dayan_wait_vec();
                }
            }
        }
    }
    matmul_check(dayan_thread_join_all());
    for (uint64_t col = 0; col < config->down.n; col += l.columns) {
        const int columns = extent(config->down.n - col, l.columns);
        const bool lowp = config->down.output_type == kDayanDataTypeBF16;
        uint64_t source = s.physical + l.sum + col * 4;
        if (lowp) {
            normalization_quantize<Bf16Format>(s.physical + l.c[1], source, s.physical + l.c[0], columns);
            source = s.physical + l.c[1];
        }
        set_lsu_matrix<0>(checksum, 1, columns * (lowp ? 2 : 4), 0, 0);
        store_matrix<0>(columns * (lowp ? 2 : 4), 1, columns * (lowp ? 2 : 4), 1, source,
                        (uintptr_t) output + col * (lowp ? 2 : 4));
        dayan_wait_lsu();
    }
    return 0;
}

} // namespace ggml_dayan_moe_ffn_detail
#endif
