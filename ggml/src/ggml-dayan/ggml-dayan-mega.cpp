#include "ggml-dayan-mega.h"

#include "ggml-dayan-kernels.h"
#include "ggml-dayan-mxfp4-layout.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

static bool ggml_backend_dayan_mega_type_code(enum ggml_type type, uint32_t * code) {
    switch (type) {
        case GGML_TYPE_F32:  *code = kDayanDataTypeF32;  return true;
        case GGML_TYPE_BF16: *code = kDayanDataTypeBF16; return true;
        case GGML_TYPE_F16:  *code = kDayanDataTypeF16;  return true;
        default: return false;
    }
}

static DayanNormConfig ggml_backend_dayan_mega_norm_config(const ggml_tensor * rms) {
    const ggml_tensor * input  = rms->src[0];
    DayanNormConfig     config = {};
    config.width               = (uint32_t) rms->ne[0];
    config.rows                = (uint32_t) ggml_nrows(rms);
    const float inv_width      = 1.0f / (float) config.width;
    std::memcpy(&config.inv_width_bits, &inv_width, sizeof(config.inv_width_bits));
    ggml_backend_dayan_mega_type_code(input->type, &config.input_type);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.input_ne[dimension] = (uint32_t) input->ne[dimension];
        config.input_nb[dimension] = input->nb[dimension];
    }
    std::memcpy(&config.epsilon_bits, rms->op_params, sizeof(config.epsilon_bits));
    return config;
}

static const ggml_tensor * ggml_backend_dayan_mega_norm_scale(const ggml_tensor * mul, const ggml_tensor * rms) {
    return mul->src[0] == rms ? mul->src[1] : mul->src[0];
}

class ggml_backend_dayan_mega_parameters {
public:
    ggml_backend_dayan_mega_parameters(const ggml_backend_dayan_mega_runtime & runtime,
                                        uint16_t graph,
                                        uint16_t count) : runtime_(runtime), graph_(graph), count_(count) {
        data_.resize(sizeof(DayanMegaConfig) + (size_t) count * sizeof(DayanMegaParameter));
        const DayanMegaConfig header = { 0, graph, count };
        std::memcpy(data_.data(), &header, sizeof(header));
    }

    template <typename T>
    bool add_data(const T & value) {
        return add_data(&value, sizeof(value), alignof(T));
    }

    bool add_data(const void * value, size_t size, size_t alignment) {
        if (next_ >= count_ || value == nullptr || size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
            return false;
        }
        const size_t offset = (data_.size() + alignment - 1) & ~(alignment - 1);
        if (offset > UINT32_MAX || size > UINT32_MAX - offset) {
            return false;
        }
        data_.resize(offset + size);
        std::memcpy(data_.data() + offset, value, size);
        return add_parameter(0, (uint32_t) offset);
    }

    bool add_tensor(const ggml_tensor * tensor, uint64_t extra_offset = 0) {
        if (next_ >= count_ || tensor == nullptr || tensor->data == nullptr) {
            return false;
        }
        npu_buffer_t * root_buffer = nullptr;
        uint64_t       root_offset = 0;
        if (!runtime_.resolve(runtime_.context, tensor->data, 1, &root_buffer, &root_offset) ||
            root_offset > UINT32_MAX || extra_offset > UINT32_MAX - root_offset) {
            return false;
        }
        size_t root = 0;
        for (; root < root_count_ && roots_[root] != root_buffer; ++root) {}
        if (root == root_count_) {
            if (root_count_ == roots_.size()) {
                return false;
            }
            roots_[root_count_++] = root_buffer;
        }
        return add_parameter((uint16_t) root + 1, (uint32_t) (root_offset + extra_offset));
    }

    bool finish() {
        if (next_ != count_ || data_.size() > UINT32_MAX) {
            return false;
        }
        const DayanMegaConfig header = { (uint32_t) data_.size(), graph_, count_ };
        std::memcpy(data_.data(), &header, sizeof(header));
        return true;
    }

    const std::vector<uint8_t> & data() const { return data_; }
    npu_buffer_t * root(size_t index) const { return index < root_count_ ? roots_[index] : nullptr; }
    uint16_t next() const { return next_; }

private:
    bool add_parameter(uint16_t source, uint32_t offset) {
        if (next_ >= count_) {
            return false;
        }
        const DayanMegaParameter parameter = { source, 0, offset };
        std::memcpy(data_.data() + sizeof(DayanMegaConfig) + (size_t) next_ * sizeof(parameter), &parameter,
                    sizeof(parameter));
        ++next_;
        return true;
    }

    const ggml_backend_dayan_mega_runtime &        runtime_;
    uint16_t                                       graph_      = 0;
    uint16_t                                       count_      = 0;
    uint16_t                                       next_       = 0;
    size_t                                         root_count_ = 0;
    std::array<npu_buffer_t *, kDayanMegaRootCount> roots_      = {};
    std::vector<uint8_t>                           data_;
};

static bool ggml_backend_dayan_mega_add_norm(ggml_backend_dayan_mega_parameters * parameters,
                                              const ggml_backend_dayan_graph_slot & slot,
                                              bool                                  residual) {
    if (slot.kind != ggml_backend_dayan_slot_kind::norm_pipeline || slot.nodes[1] == nullptr ||
        slot.nodes[2] == nullptr || (residual && slot.nodes[0] == nullptr)) {
        return false;
    }
    ggml_tensor *       add    = residual ? slot.nodes[0] : nullptr;
    ggml_tensor *       rms    = slot.nodes[1];
    ggml_tensor *       scaled = slot.nodes[2];
    ggml_tensor *       lowp   = slot.nodes[3];
    const ggml_tensor * input  = residual ? add->src[0] : rms->src[0];
    const ggml_tensor * right  = residual ? add->src[1] : input;
    const ggml_tensor * scale  = ggml_backend_dayan_mega_norm_scale(scaled, rms);
    DayanNormPipelineConfig config = {
        ggml_backend_dayan_mega_norm_config(rms),
        residual ? kDayanNormHasResidual : 0U,
        kDayanDataTypeF32,
    };
    const uint32_t output_mask = residual ? slot.output_mask : slot.output_mask & ~kDayanNormSlotOutputResidual;
    if ((output_mask & kDayanNormSlotOutputResidual) != 0U) { config.flags |= kDayanNormWriteResidual; }
    if ((output_mask & kDayanNormSlotOutputRms) != 0U) { config.flags |= kDayanNormWriteRms; }
    if ((output_mask & kDayanNormSlotOutputScaled) != 0U) { config.flags |= kDayanNormWriteScaled; }
    if ((output_mask & kDayanNormSlotOutputLowp) != 0U) {
        if (lowp == nullptr || !ggml_backend_dayan_mega_type_code(lowp->type, &config.output_type)) {
            return false;
        }
        config.flags |= kDayanNormWriteLowp;
    }
    if ((config.flags & (kDayanNormWriteRms | kDayanNormWriteScaled | kDayanNormWriteLowp)) == 0U) {
        return false;
    }
    return parameters->add_data(config) && parameters->add_tensor(input) && parameters->add_tensor(right) &&
           parameters->add_tensor(scale) && parameters->add_tensor(residual ? add : rms) &&
           parameters->add_tensor(rms) && parameters->add_tensor(scaled) &&
           parameters->add_tensor(lowp != nullptr ? lowp : scaled);
}

static bool ggml_backend_dayan_mega_add_residual(ggml_backend_dayan_mega_parameters * parameters,
                                                  const ggml_backend_dayan_graph_slot & slot,
                                                  ggml_type                             type) {
    ggml_tensor * add = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::norm_pipeline || add == nullptr || add->type != type) {
        return false;
    }
    const DayanElementwiseConfig config = { (uint64_t) ggml_nelements(add) };
    return parameters->add_data(config) && parameters->add_tensor(add->src[0]) &&
           parameters->add_tensor(add->src[1]) && parameters->add_tensor(add);
}

static bool ggml_backend_dayan_mega_add_scale(ggml_backend_dayan_mega_parameters * parameters,
                                               const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::scale || output == nullptr || output->type != GGML_TYPE_F32) {
        return false;
    }
    DayanScaleConfig config = { (uint64_t) ggml_nelements(output), 0, 0 };
    std::memcpy(&config.scale_bits, output->op_params, sizeof(config.scale_bits));
    std::memcpy(&config.bias_bits, (const char *) output->op_params + sizeof(float), sizeof(config.bias_bits));
    return parameters->add_data(config) && parameters->add_tensor(output->src[0]) && parameters->add_tensor(output);
}

// decode graphs have no build_rs state zeroing, but the fixed device parameter
// layout still expects the scale entries: pack a zero-element no-op bound to a
// tensor already present in the window (never dereferenced on device)
static bool ggml_backend_dayan_mega_add_noop_scale(ggml_backend_dayan_mega_parameters * parameters,
                                                   const ggml_tensor *                   placeholder) {
    if (placeholder == nullptr || placeholder->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanScaleConfig config = { 0, 0, 0 };
    return parameters->add_data(config) && parameters->add_tensor(placeholder) &&
           parameters->add_tensor(placeholder);
}

static bool ggml_backend_dayan_mega_add_rows(ggml_backend_dayan_mega_parameters * parameters,
                                              const ggml_backend_dayan_graph_slot & slot,
                                              bool                                  set) {
    ggml_tensor *       output  = slot.nodes[0];
    const ggml_tensor * source  = output != nullptr ? output->src[0] : nullptr;
    const ggml_tensor * indices = output != nullptr ? output->src[1] : nullptr;
    if (output == nullptr || source == nullptr || indices == nullptr ||
        slot.kind != (set ? ggml_backend_dayan_slot_kind::set_rows : ggml_backend_dayan_slot_kind::get_rows)) {
        return false;
    }
    DayanRowsConfig config = {};
    if (set) {
        if (output->src[2] == nullptr || output->data != output->src[2]->data || source->type != GGML_TYPE_F32 ||
            !ggml_backend_dayan_mega_type_code(output->type, &config.destination_type)) {
            return false;
        }
        config.row_elements     = (uint32_t) source->ne[0];
        config.rows             = (uint32_t) ggml_nrows(source);
        config.source_rows      = config.rows;
        config.destination_rows = (uint64_t) ggml_nrows(output);
        config.source_type      = kDayanDataTypeF32;
        config.flags            = kDayanRowsSet;
        for (int dimension = 0; dimension < 3; ++dimension) {
            config.index_ne[dimension]       = (uint32_t) indices->ne[dimension];
            config.source_ne[dimension]      = (uint32_t) source->ne[dimension + 1];
            config.destination_ne[dimension] = (uint32_t) output->ne[dimension + 1];
        }
    } else {
        config.row_elements     = (uint32_t) output->ne[0];
        config.rows             = (uint32_t) ggml_nrows(output);
        config.source_rows      = (uint64_t) ggml_nrows(source);
        config.destination_rows = (uint64_t) ggml_nrows(output);
        config.source_type      = kDayanDataTypeMXFP4;
        if (source->type == GGML_TYPE_MXFP4) {
            const DayanMxfp4Layout layout = DayanMxfp4Layout::make(ggml_nrows(source), source->ne[0]);
            config.source_row_stride   = layout.row_stride;
            config.source_scale_stride = layout.scale_stride;
            config.source_scale_offset = layout.scale_offset;
        } else if (!ggml_backend_dayan_mega_type_code(source->type, &config.source_type)) {
            return false;
        }
        if (!ggml_backend_dayan_mega_type_code(output->type, &config.destination_type)) {
            return false;
        }
        for (int dimension = 0; dimension < 3; ++dimension) {
            config.index_ne[dimension]       = (uint32_t) indices->ne[dimension];
            config.index_nb[dimension]       = indices->nb[dimension];
            config.source_ne[dimension]      = (uint32_t) source->ne[dimension + 1];
            config.destination_ne[dimension] = (uint32_t) output->ne[dimension + 1];
        }
    }
    config.index_type = indices->type == GGML_TYPE_I32 ? kDayanDataTypeI32 : kDayanDataTypeI64;
    return parameters->add_data(config) && parameters->add_tensor(source) && parameters->add_tensor(indices) &&
           parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_quantize(ggml_backend_dayan_mega_parameters * parameters,
                                                  const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * input  = output != nullptr ? output->src[0] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::copy || output == nullptr || input == nullptr ||
        output->type != GGML_TYPE_MXFP4) {
        return false;
    }
    const DayanMxfp4Layout layout = DayanMxfp4Layout::make(ggml_nrows(output), output->ne[0]);
    DayanMxfp4Config config = {};
    config.m                   = layout.rows;
    config.n                   = layout.columns;
    config.a_stride            = input->nb[1];
    config.output_stride       = layout.row_stride;
    config.output_scale_stride = layout.scale_stride;
    if (!ggml_backend_dayan_mega_type_code(input->type, &config.source_type)) {
        return false;
    }
    return parameters->add_data(config) && parameters->add_tensor(input) && parameters->add_tensor(output) &&
           parameters->add_tensor(output, layout.scale_offset);
}

static bool ggml_backend_dayan_mega_add_matmul(ggml_backend_dayan_mega_parameters * parameters,
                                                const ggml_tensor *                  node,
                                                const ggml_tensor *                  up,
                                                ggml_tensor *                        output,
                                                uint32_t                             operation) {
    if (node == nullptr || output == nullptr || node->src[0] == nullptr || node->src[1] == nullptr ||
        node->src[0]->type != GGML_TYPE_MXFP4 || node->src[1]->type != GGML_TYPE_MXFP4 ||
        (up != nullptr && (up->src[0] == nullptr || up->src[0]->type != GGML_TYPE_MXFP4))) {
        return false;
    }
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * input  = node->src[1];
    const ggml_tensor * up_weight = up != nullptr ? up->src[0] : weight;
    const DayanMxfp4Layout a = DayanMxfp4Layout::make(ggml_nrows(input), input->ne[0]);
    const DayanMxfp4Layout b = DayanMxfp4Layout::make(ggml_nrows(weight), weight->ne[0]);
    const DayanMxfp4Layout u = DayanMxfp4Layout::make(ggml_nrows(up_weight), up_weight->ne[0]);
    const bool output_mxfp4 = output->type == GGML_TYPE_MXFP4;
    const DayanMxfp4Layout out = DayanMxfp4Layout::make(ggml_nrows(output), output->ne[0]);
    if (output->type != GGML_TYPE_F32 && !output_mxfp4) {
        return false;
    }
    DayanMxfp4Config config = {};
    config.m                   = ggml_nrows(input);
    config.n                   = weight->ne[1];
    config.k                   = weight->ne[0];
    config.a_stride            = a.row_stride;
    config.b_stride            = b.row_stride;
    config.output_stride       = output_mxfp4 ? out.row_stride : output->nb[1];
    config.a_scale_stride      = a.scale_stride;
    config.b_scale_stride      = b.scale_stride;
    config.output_scale_stride = output_mxfp4 ? out.scale_stride : 0;
    config.source_type         = kDayanDataTypeMXFP4;
    config.weight_type         = kDayanDataTypeMXFP4;
    config.output_type         = output_mxfp4 ? kDayanDataTypeMXFP4 : kDayanDataTypeF32;
    config.operation           = operation;
    return parameters->add_data(config) && parameters->add_tensor(input) &&
           parameters->add_tensor(input, a.scale_offset) && parameters->add_tensor(weight) &&
           parameters->add_tensor(weight, b.scale_offset) && parameters->add_tensor(up_weight) &&
           parameters->add_tensor(up_weight, u.scale_offset) && parameters->add_tensor(output) &&
           parameters->add_tensor(output, output_mxfp4 ? out.scale_offset : 0);
}

static DayanMatmulConfig ggml_backend_dayan_mega_matmul_config(const ggml_tensor * weight,
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

static bool ggml_backend_dayan_mega_add_bf16_matmul(ggml_backend_dayan_mega_parameters * parameters,
                                                     const ggml_tensor *                  node,
                                                     ggml_tensor *                        output) {
    const ggml_tensor * weight = node != nullptr ? node->src[0] : nullptr;
    const ggml_tensor * input  = node != nullptr ? node->src[1] : nullptr;
    if (weight == nullptr || input == nullptr || output == nullptr || weight->type != GGML_TYPE_BF16 ||
        input->type != GGML_TYPE_BF16 || output->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanMatmulConfig config = ggml_backend_dayan_mega_matmul_config(weight, input, output);
    return parameters->add_data(config) && parameters->add_tensor(input) && parameters->add_tensor(weight) &&
           parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_bf16_matmul_pair(ggml_backend_dayan_mega_parameters * parameters,
                                                          const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * first  = slot.nodes[0];
    const ggml_tensor * second = slot.nodes[1];
    const ggml_tensor * input  = first != nullptr ? first->src[1] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::mul_mat_pair || first == nullptr || second == nullptr ||
        input == nullptr || first->src[0] == nullptr || second->src[0] == nullptr || second->src[1] != input ||
        input->type != GGML_TYPE_BF16 || first->src[0]->type != GGML_TYPE_BF16 ||
        second->src[0]->type != GGML_TYPE_BF16 || first->type != GGML_TYPE_F32 || second->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanMatmulPairConfig config = {
        ggml_backend_dayan_mega_matmul_config(first->src[0], input, first),
        ggml_backend_dayan_mega_matmul_config(second->src[0], input, second),
    };
    return parameters->add_data(config) && parameters->add_tensor(input) && parameters->add_tensor(first->src[0]) &&
           parameters->add_tensor(second->src[0]) && parameters->add_tensor(first) && parameters->add_tensor(second);
}

static bool ggml_backend_dayan_mega_add_bf16_swiglu(ggml_backend_dayan_mega_parameters * parameters,
                                                     const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * gate   = slot.nodes[0];
    const ggml_tensor * up     = slot.nodes[1];
    ggml_tensor *       output = slot.nodes[3];
    const ggml_tensor * input  = gate != nullptr ? gate->src[1] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::mul_mat_swiglu || gate == nullptr || up == nullptr ||
        output == nullptr || input == nullptr || gate->src[0] == nullptr || up->src[0] == nullptr ||
        up->src[1] != input || input->type != GGML_TYPE_BF16 || gate->src[0]->type != GGML_TYPE_BF16 ||
        up->src[0]->type != GGML_TYPE_BF16 || output->type != GGML_TYPE_BF16) {
        return false;
    }
    const DayanMatmulConfig config = ggml_backend_dayan_mega_matmul_config(gate->src[0], input, output);
    return parameters->add_data(config) && parameters->add_tensor(input) && parameters->add_tensor(gate->src[0]) &&
           parameters->add_tensor(up->src[0]) && parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_bf16_quantized(ggml_backend_dayan_mega_parameters * parameters,
                                                        const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * node   = slot.nodes[0];
    ggml_tensor *       output = slot.nodes[1];
    const ggml_tensor * weight = node != nullptr ? node->src[0] : nullptr;
    const ggml_tensor * input  = node != nullptr ? node->src[1] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::mul_mat_quantized || weight == nullptr || input == nullptr ||
        output == nullptr || weight->type != GGML_TYPE_BF16 || input->type != GGML_TYPE_BF16 ||
        output->type != GGML_TYPE_BF16) {
        return false;
    }
    const DayanMatmulConfig config = ggml_backend_dayan_mega_matmul_config(weight, input, output);
    return parameters->add_data(config) && parameters->add_tensor(input) && parameters->add_tensor(weight) &&
           parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_concat(ggml_backend_dayan_mega_parameters * parameters,
                                                const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * left   = output != nullptr ? output->src[0] : nullptr;
    const ggml_tensor * right  = output != nullptr ? output->src[1] : nullptr;
    DayanConcatConfig config = {};
    if (slot.kind != ggml_backend_dayan_slot_kind::concat || output == nullptr || left == nullptr || right == nullptr ||
        !ggml_backend_dayan_mega_type_code(output->type, &config.type)) {
        return false;
    }
    config.axis = (uint32_t) ggml_get_op_params_i32(output, 0);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.left_ne[dimension]   = (uint32_t) left->ne[dimension];
        config.right_ne[dimension]  = (uint32_t) right->ne[dimension];
        config.output_ne[dimension] = (uint32_t) output->ne[dimension];
        config.left_nb[dimension]   = left->nb[dimension];
        config.right_nb[dimension]  = right->nb[dimension];
        config.output_nb[dimension] = output->nb[dimension];
    }
    return parameters->add_data(config) && parameters->add_tensor(left) && parameters->add_tensor(right) &&
           parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_copy(ggml_backend_dayan_mega_parameters * parameters,
                                              const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * source = output != nullptr ? output->src[0] : nullptr;
    DayanCopyConfig config = {};
    if (slot.kind != ggml_backend_dayan_slot_kind::copy || output == nullptr || source == nullptr ||
        output->type == GGML_TYPE_MXFP4 || !ggml_backend_dayan_mega_type_code(source->type, &config.source_type) ||
        !ggml_backend_dayan_mega_type_code(output->type, &config.destination_type)) {
        return false;
    }
    config.elements = (uint64_t) ggml_nelements(output);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.source_ne[dimension] = (uint32_t) source->ne[dimension];
        config.source_nb[dimension] = source->nb[dimension];
    }
    return parameters->add_data(config) && parameters->add_tensor(source) && parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_ssm_conv(ggml_backend_dayan_mega_parameters * parameters,
                                                  const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * conv   = slot.nodes[0];
    ggml_tensor *       output = slot.nodes[1];
    const ggml_tensor * input  = conv != nullptr ? conv->src[0] : nullptr;
    const ggml_tensor * weight = conv != nullptr ? conv->src[1] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::ssm_conv_silu || conv == nullptr || output == nullptr ||
        input == nullptr || weight == nullptr) {
        return false;
    }
    DayanSSMConvConfig config = {
        (uint32_t) input->ne[0], (uint32_t) input->ne[1], (uint32_t) output->ne[1],
        (uint32_t) input->ne[2], (uint32_t) weight->ne[0], 1U, 0, 0,
    };
    const float neg_log2e = -1.4426950408889634f;
    const float one       = 1.0f;
    std::memcpy(&config.neg_log2e_bits, &neg_log2e, sizeof(neg_log2e));
    std::memcpy(&config.one_bits, &one, sizeof(one));
    return parameters->add_data(config) && parameters->add_tensor(input) && parameters->add_tensor(weight) &&
           parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_l2_norm(ggml_backend_dayan_mega_parameters * parameters,
                                                 const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::l2_norm || output == nullptr || output->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanNormConfig config = ggml_backend_dayan_mega_norm_config(output);
    return parameters->add_data(config) && parameters->add_tensor(output->src[0]) && parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_binary(ggml_backend_dayan_mega_parameters * parameters,
                                                const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (output == nullptr || (slot.kind != ggml_backend_dayan_slot_kind::elementwise_add &&
                              slot.kind != ggml_backend_dayan_slot_kind::residual_add &&
                              slot.kind != ggml_backend_dayan_slot_kind::elementwise_mul) ||
        output->type != GGML_TYPE_F32) {
        return false;
    }
    // The full (output-shaped) operand is the left input; the other one may
    // broadcast. All supported kinds are commutative, so swap when needed.
    const ggml_tensor * left = output->src[0];
    const ggml_tensor * right = output->src[1];
    if (!ggml_are_same_shape(output, left)) {
        const ggml_tensor * swap = left;
        left = right;
        right = swap;
    }
    if (!ggml_are_same_shape(output, left)) {
        return false;
    }
    DayanBinaryConfig config = {};
    config.elements = (uint64_t) ggml_nelements(output);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (right->ne[dimension] <= 0 || output->ne[dimension] % right->ne[dimension] != 0) {
            return false;
        }
        config.ne[dimension] = (uint32_t) output->ne[dimension];
        config.right_ne[dimension] = (uint32_t) right->ne[dimension];
    }
    if (config.right_ne[0] != config.ne[0] && (64U % config.right_ne[0]) != 0U) {
        return false;
    }
    return parameters->add_data(config) && parameters->add_tensor(left) && parameters->add_tensor(right) &&
           parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_unary(ggml_backend_dayan_mega_parameters * parameters,
                                               const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (output == nullptr || (slot.kind != ggml_backend_dayan_slot_kind::softplus &&
                              slot.kind != ggml_backend_dayan_slot_kind::sigmoid &&
                              slot.kind != ggml_backend_dayan_slot_kind::silu) ||
        output->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanElementwiseConfig config = { (uint64_t) ggml_nelements(output) };
    return parameters->add_data(config) && parameters->add_tensor(output->src[0]) && parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_gdn(ggml_backend_dayan_mega_parameters * parameters,
                                             const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::gated_delta_net || output == nullptr) {
        return false;
    }
    const ggml_tensor * q     = output->src[0];
    const ggml_tensor * k     = output->src[1];
    const ggml_tensor * v     = output->src[2];
    const ggml_tensor * g     = output->src[3];
    const ggml_tensor * beta  = output->src[4];
    const ggml_tensor * state = output->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr || state == nullptr) {
        return false;
    }
    DayanGatedDeltaNetConfig config = {};
    config.head_dim     = (uint32_t) v->ne[0];
    config.heads        = (uint32_t) v->ne[1];
    config.tokens       = (uint32_t) v->ne[2];
    config.sequences    = (uint32_t) v->ne[3];
    config.snapshots    = (uint32_t) ggml_get_op_params_i32(output, 0);
    config.gate_width   = (uint32_t) g->ne[0];
    config.q_heads      = (uint32_t) q->ne[1];
    config.k_heads      = (uint32_t) k->ne[1];
    config.q_sequences  = (uint32_t) q->ne[3];
    config.k_sequences  = (uint32_t) k->ne[3];
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.q_nb[dimension]    = q->nb[dimension];
        config.k_nb[dimension]    = k->nb[dimension];
        config.v_nb[dimension]    = v->nb[dimension];
        config.g_nb[dimension]    = g->nb[dimension];
        config.beta_nb[dimension] = beta->nb[dimension];
    }
    const float scale = 1.0f / std::sqrt((float) config.head_dim);
    std::memcpy(&config.scale_bits, &scale, sizeof(scale));
    return parameters->add_data(config) && parameters->add_tensor(q) && parameters->add_tensor(k) &&
           parameters->add_tensor(v) && parameters->add_tensor(g) && parameters->add_tensor(beta) &&
           parameters->add_tensor(state) && parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_rms_norm(ggml_backend_dayan_mega_parameters * parameters,
                                                  const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::rms_norm || output == nullptr || output->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanNormConfig config = ggml_backend_dayan_mega_norm_config(output);
    return parameters->add_data(config) && parameters->add_tensor(output->src[0]) && parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_rope(ggml_backend_dayan_mega_parameters * parameters,
                                              const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output    = slot.nodes[0];
    const ggml_tensor * input     = output != nullptr ? output->src[0] : nullptr;
    const ggml_tensor * positions = output != nullptr ? output->src[1] : nullptr;
    const ggml_tensor * factors   = output != nullptr ? output->src[2] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::rope || output == nullptr || input == nullptr ||
        positions == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        positions->type != GGML_TYPE_I32) {
        return false;
    }
    const int32_t n_dims = ggml_get_op_params_i32(output, 1);
    const int32_t mode   = ggml_get_op_params_i32(output, 2);
    if (n_dims <= 0 || (uint32_t) (n_dims / 2) > DAYAN_ROPE_MAX_PAIRS) {
        return false;
    }
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
    if (!parameters->add_data(config) || !parameters->add_tensor(input) || !parameters->add_tensor(positions)) {
        return false;
    }
    if (factors != nullptr) {
        if (!parameters->add_tensor(factors)) {
            return false;
        }
    } else {
        const std::vector<float> ones((size_t) n_dims / 2, 1.0f);
        if (!parameters->add_data(ones.data(), ones.size() * sizeof(float), alignof(float))) {
            return false;
        }
    }
    return parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_flash_attention(ggml_backend_dayan_mega_parameters * parameters,
                                                         const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor *       output = slot.nodes[0];
    const ggml_tensor * q      = output != nullptr ? output->src[0] : nullptr;
    const ggml_tensor * k      = output != nullptr ? output->src[1] : nullptr;
    const ggml_tensor * v      = output != nullptr ? output->src[2] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::flash_attention || output == nullptr || q == nullptr ||
        k == nullptr || v == nullptr || q->type != GGML_TYPE_F16 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || output->type != GGML_TYPE_F32) {
        return false;
    }
    uint32_t causal_offset = 0;
    uint32_t kv_len        = 0;
    if (!ggml_backend_dayan_flash_attention_lengths(output, &causal_offset, &kv_len)) {
        return false;
    }
    DayanFlashAttentionConfig config = {
        (uint32_t) q->ne[3], (uint32_t) q->ne[2], (uint32_t) k->ne[2], (uint32_t) q->ne[1], kv_len,
        (uint32_t) q->ne[0], 0, causal_offset, 0, q->nb[1], q->nb[2], q->nb[3], k->nb[1], k->nb[2],
        k->nb[3], v->nb[1], v->nb[2], v->nb[3],
    };
    std::memcpy(&config.scale_bits, output->op_params, sizeof(config.scale_bits));
    if (output->src[3] != nullptr) { config.flags = 1U; }
    return parameters->add_data(config) && parameters->add_tensor(q) && parameters->add_tensor(k) &&
           parameters->add_tensor(v) && parameters->add_tensor(output);
}

static bool ggml_backend_dayan_mega_add_bf16_gated_projection(
    ggml_backend_dayan_mega_parameters * parameters,
    const ggml_backend_dayan_graph_slot & slot,
    const ggml_tensor *                   scratch) {
    const ggml_tensor * sigmoid   = slot.nodes[0];
    const ggml_tensor * gated     = slot.nodes[1];
    ggml_tensor *       output    = slot.nodes[2];
    const ggml_tensor * qg        = slot.nodes[3];
    const ggml_tensor * gate_view = sigmoid != nullptr && sigmoid->src[0] != nullptr ? sigmoid->src[0]->src[0] : nullptr;
    const ggml_tensor * attention = gated != nullptr && gated->src[0] == sigmoid ? gated->src[1] :
                                    gated != nullptr ? gated->src[0] : nullptr;
    const ggml_tensor * weight    = output != nullptr ? output->src[0] : nullptr;
    if (slot.kind != ggml_backend_dayan_slot_kind::gated_output_projection || gate_view == nullptr ||
        attention == nullptr || weight == nullptr || qg == nullptr || qg->src[1] == nullptr || scratch == nullptr ||
        attention->type != GGML_TYPE_F32 || qg->type != GGML_TYPE_F32 || weight->type != GGML_TYPE_BF16 ||
        output->type != GGML_TYPE_F32 || (scratch->type != GGML_TYPE_F16 && scratch->type != GGML_TYPE_BF16)) {
        return false;
    }
    const uint64_t tokens   = attention->ne[1];
    const uint64_t heads    = gate_view->ne[1];
    const uint64_t head_dim = gate_view->ne[0];
    const uint64_t width    = heads * head_dim;
    if (tokens > UINT32_MAX || heads > UINT32_MAX || head_dim > UINT32_MAX ||
        width > SIZE_MAX / sizeof(uint16_t) || tokens > SIZE_MAX / (width * sizeof(uint16_t)) ||
        ggml_nbytes(scratch) < (size_t) tokens * width * sizeof(uint16_t)) {
        return false;
    }
    const DayanGatedProjectionConfig config = {
        ggml_backend_dayan_mega_matmul_config(weight, qg->src[1], output),
        (uint32_t) tokens,
        (uint32_t) heads,
        (uint32_t) head_dim,
    };
    return parameters->add_data(config) && parameters->add_tensor(attention) && parameters->add_tensor(qg) &&
           parameters->add_tensor(weight) && parameters->add_tensor(output) && parameters->add_tensor(scratch);
}

static constexpr ggml_backend_dayan_slot_kind ggml_backend_dayan_qwen35_mxfp4_linear_signature[] = {
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::scale,
    ggml_backend_dayan_slot_kind::get_rows,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::concat,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::scale,
    ggml_backend_dayan_slot_kind::get_rows,
    ggml_backend_dayan_slot_kind::ssm_conv_silu,
    ggml_backend_dayan_slot_kind::l2_norm,
    ggml_backend_dayan_slot_kind::l2_norm,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::elementwise_add,
    ggml_backend_dayan_slot_kind::softplus,
    ggml_backend_dayan_slot_kind::elementwise_mul,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::sigmoid,
    ggml_backend_dayan_slot_kind::gated_delta_net,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::silu,
    ggml_backend_dayan_slot_kind::elementwise_mul,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat_swiglu,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::norm_pipeline,
};

static constexpr ggml_backend_dayan_slot_kind ggml_backend_dayan_qwen35_mxfp4_full_signature[] = {
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::rms_norm,
    ggml_backend_dayan_slot_kind::elementwise_mul,
    ggml_backend_dayan_slot_kind::rope,
    ggml_backend_dayan_slot_kind::mul_mat_pair,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::rope,
    ggml_backend_dayan_slot_kind::set_rows,
    ggml_backend_dayan_slot_kind::set_rows,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::flash_attention,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::sigmoid,
    ggml_backend_dayan_slot_kind::elementwise_mul,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat_swiglu,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::norm_pipeline,
};

static constexpr ggml_backend_dayan_slot_kind ggml_backend_dayan_qwen35_bf16_linear_signature[] = {
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::scale,
    ggml_backend_dayan_slot_kind::get_rows,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::concat,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::scale,
    ggml_backend_dayan_slot_kind::get_rows,
    ggml_backend_dayan_slot_kind::ssm_conv_silu,
    ggml_backend_dayan_slot_kind::l2_norm,
    ggml_backend_dayan_slot_kind::l2_norm,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::elementwise_add,
    ggml_backend_dayan_slot_kind::softplus,
    ggml_backend_dayan_slot_kind::elementwise_mul,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::sigmoid,
    ggml_backend_dayan_slot_kind::gated_delta_net,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::silu,
    ggml_backend_dayan_slot_kind::elementwise_mul,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::mul_mat_swiglu,
    ggml_backend_dayan_slot_kind::mul_mat_quantized,
    ggml_backend_dayan_slot_kind::norm_pipeline,
};

static constexpr ggml_backend_dayan_slot_kind ggml_backend_dayan_qwen35_bf16_full_signature[] = {
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::mul_mat,
    ggml_backend_dayan_slot_kind::rms_norm,
    ggml_backend_dayan_slot_kind::elementwise_mul,
    ggml_backend_dayan_slot_kind::rope,
    ggml_backend_dayan_slot_kind::mul_mat_pair,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::rope,
    ggml_backend_dayan_slot_kind::set_rows,
    ggml_backend_dayan_slot_kind::set_rows,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::flash_attention,
    ggml_backend_dayan_slot_kind::gated_output_projection,
    ggml_backend_dayan_slot_kind::copy,
    ggml_backend_dayan_slot_kind::norm_pipeline,
    ggml_backend_dayan_slot_kind::mul_mat_swiglu,
    ggml_backend_dayan_slot_kind::mul_mat_quantized,
    ggml_backend_dayan_slot_kind::norm_pipeline,
};

// decode graphs drop the two zero-element build_rs state scales, so the linear
// signatures also match a shorter decode variant with those positions removed
template <typename T, size_t N, size_t M>
constexpr std::array<T, N - M> ggml_backend_dayan_strip_signature(const T (&signature)[N],
                                                                  const size_t (&omitted)[M]) {
    std::array<T, N - M> stripped = {};
    size_t               out      = 0;
    for (size_t index = 0; index < N; ++index) {
        bool skip = false;
        for (size_t position = 0; position < M; ++position) {
            skip = skip || index == omitted[position];
        }
        if (!skip) {
            stripped[out++] = signature[index];
        }
    }
    return stripped;
}

static constexpr size_t ggml_backend_dayan_qwen35_linear_scale_positions[2] = { 1, 7 };

static constexpr auto ggml_backend_dayan_qwen35_mxfp4_linear_decode_signature =
    ggml_backend_dayan_strip_signature(ggml_backend_dayan_qwen35_mxfp4_linear_signature,
                                       ggml_backend_dayan_qwen35_linear_scale_positions);
static constexpr auto ggml_backend_dayan_qwen35_bf16_linear_decode_signature =
    ggml_backend_dayan_strip_signature(ggml_backend_dayan_qwen35_bf16_linear_signature,
                                       ggml_backend_dayan_qwen35_linear_scale_positions);

static bool ggml_backend_dayan_match_signature(const std::vector<ggml_backend_dayan_graph_slot> & slots,
                                                size_t                                              begin,
                                                const ggml_backend_dayan_slot_kind *                signature,
                                                size_t                                              count,
                                                size_t                                              add_position) {
    if (begin + count > slots.size()) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        const auto actual = slots[begin + index].kind;
        if (index == add_position &&
            (actual == ggml_backend_dayan_slot_kind::elementwise_add ||
             actual == ggml_backend_dayan_slot_kind::residual_add)) {
            continue;
        }
        if (actual != signature[index]) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_dayan_pack_qwen35_mxfp4_linear(ggml_backend_dayan_mega_parameters * parameters,
                                                         const std::vector<ggml_backend_dayan_graph_slot> & slots,
                                                         size_t                                              begin,
                                                         bool                                                decode) {
    const auto & s = slots;
    // decode omits the scale slots at signature positions 1 and 7, so later
    // signature positions shift down by one or two slots
    const auto at = [&s, begin, decode](size_t index) -> const ggml_backend_dayan_graph_slot & {
        const size_t skip = decode ? (index > 7 ? 2 : (index > 1 ? 1 : 0)) : 0;
        return s[begin + index - skip];
    };
    const ggml_tensor * noop_scale = decode ? s[begin + 1].nodes[0] : nullptr;
    return ggml_backend_dayan_mega_add_norm(parameters, at(0), false) &&
           (decode ? ggml_backend_dayan_mega_add_noop_scale(parameters, noop_scale)
                   : ggml_backend_dayan_mega_add_scale(parameters, at(1))) &&
           ggml_backend_dayan_mega_add_rows(parameters, at(2), false) &&
           ggml_backend_dayan_mega_add_quantize(parameters, at(3)) &&
           ggml_backend_dayan_mega_add_matmul(parameters, at(4).nodes[0], nullptr,
                                               at(4).nodes[0], 0) &&
           ggml_backend_dayan_mega_add_concat(parameters, at(5)) &&
           ggml_backend_dayan_mega_add_copy(parameters, at(6)) &&
           (decode ? ggml_backend_dayan_mega_add_noop_scale(parameters, noop_scale)
                   : ggml_backend_dayan_mega_add_scale(parameters, at(7))) &&
           ggml_backend_dayan_mega_add_rows(parameters, at(8), false) &&
           ggml_backend_dayan_mega_add_ssm_conv(parameters, at(9)) &&
           ggml_backend_dayan_mega_add_l2_norm(parameters, at(10)) &&
           ggml_backend_dayan_mega_add_l2_norm(parameters, at(11)) &&
           ggml_backend_dayan_mega_add_matmul(parameters, at(12).nodes[0], nullptr,
                                               at(12).nodes[0], 0) &&
           ggml_backend_dayan_mega_add_binary(parameters, at(13)) &&
           ggml_backend_dayan_mega_add_unary(parameters, at(14)) &&
           ggml_backend_dayan_mega_add_binary(parameters, at(15)) &&
           ggml_backend_dayan_mega_add_matmul(parameters, at(16).nodes[0], nullptr,
                                               at(16).nodes[0], 0) &&
           ggml_backend_dayan_mega_add_unary(parameters, at(17)) &&
           ggml_backend_dayan_mega_add_gdn(parameters, at(18)) &&
           ggml_backend_dayan_mega_add_copy(parameters, at(19)) &&
           ggml_backend_dayan_mega_add_norm(parameters, at(20), false) &&
           ggml_backend_dayan_mega_add_matmul(parameters, at(21).nodes[0], nullptr,
                                               at(21).nodes[0], 0) &&
           ggml_backend_dayan_mega_add_unary(parameters, at(22)) &&
           ggml_backend_dayan_mega_add_binary(parameters, at(23)) &&
           ggml_backend_dayan_mega_add_quantize(parameters, at(24)) &&
           ggml_backend_dayan_mega_add_matmul(parameters, at(25).nodes[0], nullptr,
                                               at(25).nodes[0], 0) &&
           ggml_backend_dayan_mega_add_norm(parameters, at(26), true) &&
           ggml_backend_dayan_mega_add_quantize(parameters, at(27)) &&
           ggml_backend_dayan_mega_add_matmul(parameters, at(28).nodes[0], at(28).nodes[1],
                                               at(28).nodes[3], 2) &&
           ggml_backend_dayan_mega_add_matmul(parameters, at(29).nodes[0], nullptr,
                                               at(29).nodes[0], 0) &&
           ggml_backend_dayan_mega_add_residual(parameters, at(30), GGML_TYPE_F32);
}

static bool ggml_backend_dayan_pack_qwen35_mxfp4_full(ggml_backend_dayan_mega_parameters * parameters,
                                                       const std::vector<ggml_backend_dayan_graph_slot> & slots,
                                                       size_t begin) {
    const auto & s = slots;
    return ggml_backend_dayan_mega_add_norm(parameters, s[begin + 0], false) &&
           ggml_backend_dayan_mega_add_quantize(parameters, s[begin + 1]) &&
           ggml_backend_dayan_mega_add_matmul(parameters, s[begin + 2].nodes[0], nullptr,
                                               s[begin + 2].nodes[0], 0) &&
           ggml_backend_dayan_mega_add_rms_norm(parameters, s[begin + 3]) &&
           ggml_backend_dayan_mega_add_binary(parameters, s[begin + 4]) &&
           ggml_backend_dayan_mega_add_rope(parameters, s[begin + 5]) &&
           ggml_backend_dayan_mega_add_matmul(parameters, s[begin + 6].nodes[0], nullptr,
                                               s[begin + 6].nodes[0], 0) &&
           ggml_backend_dayan_mega_add_matmul(parameters, s[begin + 6].nodes[1], nullptr,
                                               s[begin + 6].nodes[1], 0) &&
           ggml_backend_dayan_mega_add_norm(parameters, s[begin + 7], false) &&
           ggml_backend_dayan_mega_add_rope(parameters, s[begin + 8]) &&
           ggml_backend_dayan_mega_add_rows(parameters, s[begin + 9], true) &&
           ggml_backend_dayan_mega_add_rows(parameters, s[begin + 10], true) &&
           ggml_backend_dayan_mega_add_copy(parameters, s[begin + 11]) &&
           ggml_backend_dayan_mega_add_flash_attention(parameters, s[begin + 12]) &&
           ggml_backend_dayan_mega_add_copy(parameters, s[begin + 13]) &&
           ggml_backend_dayan_mega_add_unary(parameters, s[begin + 14]) &&
           ggml_backend_dayan_mega_add_binary(parameters, s[begin + 15]) &&
           ggml_backend_dayan_mega_add_quantize(parameters, s[begin + 16]) &&
           ggml_backend_dayan_mega_add_matmul(parameters, s[begin + 17].nodes[0], nullptr,
                                               s[begin + 17].nodes[0], 0) &&
           ggml_backend_dayan_mega_add_norm(parameters, s[begin + 18], true) &&
           ggml_backend_dayan_mega_add_quantize(parameters, s[begin + 19]) &&
           ggml_backend_dayan_mega_add_matmul(parameters, s[begin + 20].nodes[0], s[begin + 20].nodes[1],
                                               s[begin + 20].nodes[3], 2) &&
           ggml_backend_dayan_mega_add_matmul(parameters, s[begin + 21].nodes[0], nullptr,
                                               s[begin + 21].nodes[0], 0) &&
           ggml_backend_dayan_mega_add_residual(parameters, s[begin + 22], GGML_TYPE_F32);
}

static bool ggml_backend_dayan_pack_qwen35_bf16_linear(ggml_backend_dayan_mega_parameters * parameters,
                                                        const std::vector<ggml_backend_dayan_graph_slot> & slots,
                                                        size_t                                              begin,
                                                        bool                                                decode) {
    const auto & s = slots;
    // decode omits the scale slots at signature positions 1 and 7, so later
    // signature positions shift down by one or two slots
    const auto at = [&s, begin, decode](size_t index) -> const ggml_backend_dayan_graph_slot & {
        const size_t skip = decode ? (index > 7 ? 2 : (index > 1 ? 1 : 0)) : 0;
        return s[begin + index - skip];
    };
    const ggml_tensor * noop_scale = decode ? s[begin + 1].nodes[0] : nullptr;
    return ggml_backend_dayan_mega_add_norm(parameters, at(0), false) &&
           (decode ? ggml_backend_dayan_mega_add_noop_scale(parameters, noop_scale)
                   : ggml_backend_dayan_mega_add_scale(parameters, at(1))) &&
           ggml_backend_dayan_mega_add_rows(parameters, at(2), false) &&
           ggml_backend_dayan_mega_add_copy(parameters, at(3)) &&
           ggml_backend_dayan_mega_add_bf16_matmul(parameters, at(4).nodes[0], at(4).nodes[0]) &&
           ggml_backend_dayan_mega_add_concat(parameters, at(5)) &&
           ggml_backend_dayan_mega_add_copy(parameters, at(6)) &&
           (decode ? ggml_backend_dayan_mega_add_noop_scale(parameters, noop_scale)
                   : ggml_backend_dayan_mega_add_scale(parameters, at(7))) &&
           ggml_backend_dayan_mega_add_rows(parameters, at(8), false) &&
           ggml_backend_dayan_mega_add_ssm_conv(parameters, at(9)) &&
           ggml_backend_dayan_mega_add_l2_norm(parameters, at(10)) &&
           ggml_backend_dayan_mega_add_l2_norm(parameters, at(11)) &&
           ggml_backend_dayan_mega_add_bf16_matmul(parameters, at(12).nodes[0], at(12).nodes[0]) &&
           ggml_backend_dayan_mega_add_binary(parameters, at(13)) &&
           ggml_backend_dayan_mega_add_unary(parameters, at(14)) &&
           ggml_backend_dayan_mega_add_binary(parameters, at(15)) &&
           ggml_backend_dayan_mega_add_bf16_matmul(parameters, at(16).nodes[0], at(16).nodes[0]) &&
           ggml_backend_dayan_mega_add_unary(parameters, at(17)) &&
           ggml_backend_dayan_mega_add_gdn(parameters, at(18)) &&
           ggml_backend_dayan_mega_add_copy(parameters, at(19)) &&
           ggml_backend_dayan_mega_add_norm(parameters, at(20), false) &&
           ggml_backend_dayan_mega_add_bf16_matmul(parameters, at(21).nodes[0], at(21).nodes[0]) &&
           ggml_backend_dayan_mega_add_unary(parameters, at(22)) &&
           ggml_backend_dayan_mega_add_binary(parameters, at(23)) &&
           ggml_backend_dayan_mega_add_copy(parameters, at(24)) &&
           ggml_backend_dayan_mega_add_bf16_matmul(parameters, at(25).nodes[0], at(25).nodes[0]) &&
           ggml_backend_dayan_mega_add_copy(parameters, at(26)) &&
           ggml_backend_dayan_mega_add_norm(parameters, at(27), true) &&
           ggml_backend_dayan_mega_add_bf16_swiglu(parameters, at(28)) &&
           ggml_backend_dayan_mega_add_bf16_quantized(parameters, at(29)) &&
           ggml_backend_dayan_mega_add_residual(parameters, at(30), GGML_TYPE_BF16);
}

static bool ggml_backend_dayan_pack_qwen35_bf16_full(ggml_backend_dayan_mega_parameters * parameters,
                                                      const std::vector<ggml_backend_dayan_graph_slot> & slots,
                                                      size_t begin) {
    const auto & s = slots;
#define PACK_STEP(index, expression) \
    if (!(expression)) { \
        GGML_LOG_ERROR("Qwen3.5 BF16 full pack failed at slot %d (%s)\\n", index, s[begin + index].nodes[0] ? s[begin + index].nodes[0]->name : "null"); \
        return false; \
    }
    PACK_STEP(0, ggml_backend_dayan_mega_add_norm(parameters, s[begin + 0], false));
    PACK_STEP(1, ggml_backend_dayan_mega_add_bf16_matmul(parameters, s[begin + 1].nodes[0], s[begin + 1].nodes[0]));
    PACK_STEP(2, ggml_backend_dayan_mega_add_rms_norm(parameters, s[begin + 2]));
    PACK_STEP(3, ggml_backend_dayan_mega_add_binary(parameters, s[begin + 3]));
    PACK_STEP(4, ggml_backend_dayan_mega_add_rope(parameters, s[begin + 4]));
    PACK_STEP(5, ggml_backend_dayan_mega_add_bf16_matmul_pair(parameters, s[begin + 5]));
    PACK_STEP(6, ggml_backend_dayan_mega_add_norm(parameters, s[begin + 6], false));
    PACK_STEP(7, ggml_backend_dayan_mega_add_rope(parameters, s[begin + 7]));
    PACK_STEP(8, ggml_backend_dayan_mega_add_rows(parameters, s[begin + 8], true));
    PACK_STEP(9, ggml_backend_dayan_mega_add_rows(parameters, s[begin + 9], true));
    PACK_STEP(10, ggml_backend_dayan_mega_add_copy(parameters, s[begin + 10]));
    PACK_STEP(11, ggml_backend_dayan_mega_add_flash_attention(parameters, s[begin + 11]));
    PACK_STEP(12, ggml_backend_dayan_mega_add_bf16_gated_projection(parameters, s[begin + 12], s[begin + 10].nodes[0]));
    PACK_STEP(13, ggml_backend_dayan_mega_add_copy(parameters, s[begin + 13]));
    PACK_STEP(14, ggml_backend_dayan_mega_add_norm(parameters, s[begin + 14], true));
    PACK_STEP(15, ggml_backend_dayan_mega_add_bf16_swiglu(parameters, s[begin + 15]));
    PACK_STEP(16, ggml_backend_dayan_mega_add_bf16_quantized(parameters, s[begin + 16]));
    PACK_STEP(17, ggml_backend_dayan_mega_add_residual(parameters, s[begin + 17], GGML_TYPE_BF16));
#undef PACK_STEP
    return true;
}

ggml_backend_dayan_mega_result ggml_backend_dayan_try_qwen35_mega(
    const ggml_backend_dayan_mega_runtime &            runtime,
    const std::vector<ggml_backend_dayan_graph_slot> & slots,
    size_t                                              begin) {
    ggml_backend_dayan_mega_result result = { 0, begin, false };
    if (!runtime.enabled || runtime.resolve == nullptr || runtime.launch == nullptr) {
        return result;
    }
    constexpr size_t mxfp4_linear_count = sizeof(ggml_backend_dayan_qwen35_mxfp4_linear_signature) /
                                          sizeof(ggml_backend_dayan_qwen35_mxfp4_linear_signature[0]);
    constexpr size_t mxfp4_linear_decode_count = ggml_backend_dayan_qwen35_mxfp4_linear_decode_signature.size();
    constexpr size_t mxfp4_full_count = sizeof(ggml_backend_dayan_qwen35_mxfp4_full_signature) /
                                        sizeof(ggml_backend_dayan_qwen35_mxfp4_full_signature[0]);
    constexpr size_t bf16_linear_count = sizeof(ggml_backend_dayan_qwen35_bf16_linear_signature) /
                                         sizeof(ggml_backend_dayan_qwen35_bf16_linear_signature[0]);
    constexpr size_t bf16_linear_decode_count = ggml_backend_dayan_qwen35_bf16_linear_decode_signature.size();
    constexpr size_t bf16_full_count = sizeof(ggml_backend_dayan_qwen35_bf16_full_signature) /
                                       sizeof(ggml_backend_dayan_qwen35_bf16_full_signature[0]);
    constexpr size_t no_add = std::numeric_limits<size_t>::max();
    uint16_t graph = 0;
    uint16_t parameter_count = 0;
    bool decode = false;
    if (ggml_backend_dayan_match_signature(
            slots, begin, ggml_backend_dayan_qwen35_mxfp4_linear_signature, mxfp4_linear_count, 13)) {
        graph = kDayanMegaQwen35DenseMxfp4Linear;
        parameter_count = kDayanMegaQwen35DenseMxfp4LinearParameters;
        result.end = begin + mxfp4_linear_count - 1;
    } else if (ggml_backend_dayan_match_signature(
                   slots, begin, ggml_backend_dayan_qwen35_mxfp4_linear_decode_signature.data(),
                   mxfp4_linear_decode_count, 11)) {
        graph = kDayanMegaQwen35DenseMxfp4Linear;
        parameter_count = kDayanMegaQwen35DenseMxfp4LinearParameters;
        decode = true;
        result.end = begin + mxfp4_linear_decode_count - 1;
    } else if (ggml_backend_dayan_match_signature(
                   slots, begin, ggml_backend_dayan_qwen35_mxfp4_full_signature, mxfp4_full_count, no_add)) {
        graph = kDayanMegaQwen35DenseMxfp4Full;
        parameter_count = kDayanMegaQwen35DenseMxfp4FullParameters;
        result.end = begin + mxfp4_full_count - 1;
    } else if (ggml_backend_dayan_match_signature(
                   slots, begin, ggml_backend_dayan_qwen35_bf16_linear_signature, bf16_linear_count, 13)) {
        graph = kDayanMegaQwen35DenseBf16Linear;
        parameter_count = kDayanMegaQwen35DenseBf16LinearParameters;
        result.end = begin + bf16_linear_count - 1;
    } else if (ggml_backend_dayan_match_signature(
                   slots, begin, ggml_backend_dayan_qwen35_bf16_linear_decode_signature.data(),
                   bf16_linear_decode_count, 11)) {
        graph = kDayanMegaQwen35DenseBf16Linear;
        parameter_count = kDayanMegaQwen35DenseBf16LinearParameters;
        decode = true;
        result.end = begin + bf16_linear_decode_count - 1;
    } else if (ggml_backend_dayan_match_signature(
                   slots, begin, ggml_backend_dayan_qwen35_bf16_full_signature, bf16_full_count, no_add)) {
        graph = kDayanMegaQwen35DenseBf16Full;
        parameter_count = kDayanMegaQwen35DenseBf16FullParameters;
        result.end = begin + bf16_full_count - 1;
    } else {
        return result;
    }

    ggml_backend_dayan_mega_parameters parameters(runtime, graph, parameter_count);
    bool packed = false;
    switch (graph) {
        case kDayanMegaQwen35DenseMxfp4Linear:
            packed = ggml_backend_dayan_pack_qwen35_mxfp4_linear(&parameters, slots, begin, decode);
            break;
        case kDayanMegaQwen35DenseMxfp4Full:
            packed = ggml_backend_dayan_pack_qwen35_mxfp4_full(&parameters, slots, begin);
            break;
        case kDayanMegaQwen35DenseBf16Linear:
            packed = ggml_backend_dayan_pack_qwen35_bf16_linear(&parameters, slots, begin, decode);
            break;
        case kDayanMegaQwen35DenseBf16Full:
            packed = ggml_backend_dayan_pack_qwen35_bf16_full(&parameters, slots, begin);
            break;
    }
    if (!packed || !parameters.finish()) {
        result.end = begin;
        return result;
    }

    npu_buffer_t * roots[kDayanMegaRootCount] = {};
    for (size_t index = 0; index < kDayanMegaRootCount; ++index) {
        roots[index] = parameters.root(index);
    }
    result.launched = true;
    result.status = runtime.launch(runtime.context, graph, parameters.data().data(), parameters.data().size(), roots);
    return result;
}
