#include "ggml-dayan-mega.h"
#include "ggml-dayan-moe-ffn.h"

#include "ggml-dayan-kernels.h"
#include "ggml-dayan-mxfp4-layout.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace mega = ggml_dayan_mega_detail;

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

    struct tensor_ref {
        const ggml_tensor * tensor;
        uint64_t offset;
    };

    struct data_ref {
        const void * data;
        size_t size;
        size_t alignment;
    };

    template <auto Function>
    void expect() { function_ = &function_tag<Function>; }

    template <auto Function, typename Config, typename... Args>
    bool add(const Config & config, const Args &... args) {
        using signature = mega::Signature<decltype(Function)>;
        static_assert(std::is_same<Config, typename signature::config_type>::value, "Mega config type mismatch");
        static_assert(1 + sizeof...(Args) == signature::count, "Mega argument count mismatch");
        return function_ == &function_tag<Function> && add_data(config) && (add_input(args) && ...);
    }

private:
    template <auto Function> static constexpr char function_tag = 0;

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
            extra_offset > UINT64_MAX - root_offset) {
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
        return add_parameter((uint16_t) root + 1, root_offset + extra_offset);
    }

    bool add_buffer(npu_buffer_t * buffer) {
        if (next_ >= count_ || buffer == nullptr) {
            return false;
        }
        size_t root = 0;
        for (; root < root_count_ && roots_[root] != buffer; ++root) {}
        if (root == root_count_) {
            if (root_count_ == roots_.size()) {
                return false;
            }
            roots_[root_count_++] = buffer;
        }
        return add_parameter((uint16_t) root + 1, 0);
    }

public:
    bool finish() {
        if (next_ != count_ || data_.size() > UINT32_MAX) {
            return false;
        }
        const size_t workspace_offset = (data_.size() + 63) & ~size_t(63);
        if (workspace_offset > UINT32_MAX - sizeof(DayanClusterState)) { return false; }
        data_.resize(workspace_offset + sizeof(DayanClusterState));
        const DayanMegaConfig header = { (uint32_t) data_.size(), graph_, count_, (uint32_t) workspace_offset };
        std::memcpy(data_.data(), &header, sizeof(header));
        return true;
    }

    const std::vector<uint8_t> & data() const { return data_; }
    npu_buffer_t * root(size_t index) const { return index < root_count_ ? roots_[index] : nullptr; }
    uint16_t next() const { return next_; }

private:
    bool add_input(const ggml_tensor * tensor) { return add_tensor(tensor); }
    bool add_input(tensor_ref ref) { return add_tensor(ref.tensor, ref.offset); }
    bool add_input(data_ref ref) { return add_data(ref.data, ref.size, ref.alignment); }
    bool add_input(npu_buffer_t * buffer) { return add_buffer(buffer); }

    bool add_parameter(uint16_t source, uint64_t offset) {
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
    const char *                                   function_   = nullptr;
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
    return parameters->add<mega::norm_pipeline>(config, input, right, scale, residual ? add : rms, rms, scaled, lowp != nullptr ? lowp : scaled);
}

static bool ggml_backend_dayan_mega_add_residual(ggml_backend_dayan_mega_parameters * parameters,
                                                  const ggml_backend_dayan_graph_slot & slot,
                                                  ggml_type                             type) {
    ggml_tensor * add = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::norm_pipeline || add == nullptr || add->type != type) {
        return false;
    }
    const DayanElementwiseConfig config = { (uint64_t) ggml_nelements(add) };
    return type == GGML_TYPE_BF16 ?
        parameters->add<mega::residual_add_bf16>(config, add->src[0], add->src[1], add) :
        parameters->add<mega::residual_add>(config, add->src[0], add->src[1], add);
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
    return parameters->add<mega::scale>(config, output->src[0], output);
}

// Decode omits state zeroing. Keep the shared device flow with a zero-element scale.
static bool ggml_backend_dayan_mega_add_noop_scale(ggml_backend_dayan_mega_parameters * parameters,
                                                   const ggml_tensor *                   placeholder) {
    if (placeholder == nullptr || placeholder->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanScaleConfig config = { 0, 0, 0 };
    return parameters->add<mega::scale>(config, placeholder, placeholder);
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
    return set ? parameters->add<mega::set_rows>(config, source, indices, output) :
                 parameters->add<mega::get_rows>(config, source, indices, output);
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
    return parameters->add<mega::quantize_mxfp4>(config, input, output, ggml_backend_dayan_mega_parameters::tensor_ref{output, layout.scale_offset});
}

static bool ggml_backend_dayan_mega_add_matmul(ggml_backend_dayan_mega_parameters * parameters,
                                                const ggml_tensor *                  node,
                                                const ggml_tensor *                  up,
                                                ggml_tensor *                        output,
                                                uint32_t                             operation,
                                                npu_buffer_t *                       workspace = nullptr,
                                                uint64_t                             workspace_bytes = 0) {
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
    const char * share_inputs = std::getenv("GGML_DAYAN_MXFP4_SHARE_INPUTS");
    config.share_inputs = share_inputs == nullptr || std::strcmp(share_inputs, "0") != 0;
    const bool moe = node->op == GGML_OP_MUL_MAT_ID;
    if (moe && node->src[2] == nullptr) {
        return false;
    }
    DayanMxfp4MoeConfig routed = {};
    if (moe) {
        config.m = 1;
        routed.matmul = config;
        routed.routing = {(uint32_t) config.k, (uint32_t) config.n, (uint32_t) weight->ne[2],
                          (uint32_t) node->src[2]->ne[0], (uint32_t) node->src[2]->ne[1],
                          (uint32_t) input->ne[1], node->src[2]->nb[1]};
        routed.grouped = routed.routing.tokens > 1;
        routed.workspace_bytes = workspace_bytes;
        if (workspace == nullptr) {
            return false;
        }
    }
    using tensor_ref = ggml_backend_dayan_mega_parameters::tensor_ref;
    if (moe) {
        return parameters->add<mega::matmul_id_mxfp4>(routed, input, tensor_ref{input, a.scale_offset}, weight,
            tensor_ref{weight, b.scale_offset}, up_weight, tensor_ref{up_weight, u.scale_offset}, node->src[2],
            output, tensor_ref{output, output_mxfp4 ? out.scale_offset : 0}, workspace);
    }
    return parameters->add<mega::matmul_mxfp4>(config, input, tensor_ref{input, a.scale_offset}, weight,
        tensor_ref{weight, b.scale_offset}, up_weight, tensor_ref{up_weight, u.scale_offset},
        output, tensor_ref{output, output_mxfp4 ? out.scale_offset : 0});
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
    return parameters->add<mega::matmul_bf16>(config, input, weight, output);
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
    return parameters->add<mega::matmul_pair_bf16>(config, input, first->src[0], second->src[0], first, second);
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
    return parameters->add<mega::matmul_swiglu_bf16>(config, input, gate->src[0], up->src[0], output);
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
    return parameters->add<mega::matmul_quantized_bf16>(config, input, weight, output);
}

static bool ggml_backend_dayan_mega_add_noop(ggml_backend_dayan_mega_parameters *, int) { return true; }

static bool ggml_backend_dayan_mega_add_ffn(ggml_backend_dayan_mega_parameters * parameters,
                                           const ggml_backend_dayan_graph_slot & gate_slot,
                                           const ggml_backend_dayan_graph_slot & down_slot,
                                           const ggml_backend_dayan_graph_slot * weighted_slot,
                                           const ggml_backend_dayan_graph_slot * sum_slot) {
    const char * enabled = std::getenv("GGML_DAYAN_FUSED_FFN");
    if (enabled && std::strcmp(enabled, "0") == 0) { return false; }
    const auto * gate = gate_slot.nodes[0];
    const auto * up = gate_slot.nodes[1];
    const auto * hidden = gate_slot.nodes[3];
    const auto * down = down_slot.nodes[0];
    const bool moe = weighted_slot != nullptr;
    const auto * weighted = moe ? weighted_slot->nodes[0] : nullptr;
    auto * output = moe ? (sum_slot ? sum_slot->nodes[0] : nullptr) :
        down_slot.kind == ggml_backend_dayan_slot_kind::mul_mat_quantized ? down_slot.nodes[1] : down_slot.nodes[0];
    if (!gate || !up || !hidden || !down || !output || !gate->src[0] || !gate->src[1] || !up->src[0] ||
        !down->src[0] || down->src[1] != hidden || up->src[1] != gate->src[1] || ggml_nrows(gate->src[1]) != 1 ||
        (gate->src[0]->type != GGML_TYPE_MXFP4 && gate->src[0]->type != GGML_TYPE_BF16) ||
        up->src[0]->type != gate->src[0]->type || down->src[0]->type != gate->src[0]->type ||
        gate->src[1]->type != gate->src[0]->type || hidden->type != gate->src[0]->type ||
        (output->type != GGML_TYPE_F32 && output->type != GGML_TYPE_BF16) || ggml_nrows(output) != 1 ||
        gate->src[0]->ne[0] % 128 || gate->src[0]->ne[1] % 128 || down->src[0]->ne[1] % 64 ||
        down->src[0]->ne[0] != gate->src[0]->ne[1] || output->ne[0] != down->src[0]->ne[1] ||
        !ggml_is_contiguous(output) || (hidden->flags & GGML_TENSOR_FLAG_OUTPUT)) { return false; }
    const auto * input = gate->src[1];
    const auto * gw = gate->src[0];
    const auto * uw = up->src[0];
    const auto * dw = down->src[0];
    if (!ggml_is_contiguous(input) || !ggml_is_contiguous(gw) || !ggml_is_contiguous(uw) || !ggml_is_contiguous(dw) ||
        gw->ne[0] > 65536 || gw->ne[1] > 65536 || dw->ne[1] > 65536 ||
        gw->ne[0] != uw->ne[0] || gw->ne[1] != uw->ne[1] || gw->ne[2] != uw->ne[2] ||
        dw->ne[2] != gw->ne[2] || gw->ne[3] != 1 || uw->ne[3] != 1 || dw->ne[3] != 1) { return false; }
    const auto * selected = moe ? gate->src[2] : input;
    const auto * weights = moe && weighted ? weighted->src[1] : input;
    const uint32_t used = moe && selected ? selected->ne[0] : 1;
    if (moe) {
        if (!weighted || !selected || !weights || selected->type != GGML_TYPE_I32 || selected->ne[1] != 1 ||
            selected->nb[0] != sizeof(int32_t) || !used || used > 32 || used > gw->ne[2] || up->src[2] != selected || down->src[2] != selected ||
            weighted->src[0] != down || weights->type != GGML_TYPE_F32 || ggml_nelements(weights) != used ||
            !ggml_is_contiguous(weights) || output->type != GGML_TYPE_F32 ||
            (down->flags & GGML_TENSOR_FLAG_OUTPUT) || (weighted->flags & GGML_TENSOR_FLAG_OUTPUT)) { return false; }
        const ggml_tensor * accumulator = output;
        for (uint32_t e = used - 1; e > 0; --e) {
            if (accumulator->op != GGML_OP_ADD || !accumulator->src[0] || !accumulator->src[1] ||
                accumulator->src[1]->view_src != weighted || accumulator->src[1]->view_offs != e * weighted->nb[1]) { return false; }
            accumulator = accumulator->src[0];
        }
        if (accumulator->view_src != weighted || accumulator->view_offs != 0) { return false; }
    } else if (gw->ne[2] != 1) { return false; }
    const bool mx = gw->type == GGML_TYPE_MXFP4;
    const auto a = DayanMxfp4Layout::make(ggml_nrows(input), input->ne[0]);
    const auto g = DayanMxfp4Layout::make(ggml_nrows(gw), gw->ne[0]);
    const auto u = DayanMxfp4Layout::make(ggml_nrows(uw), uw->ne[0]);
    const auto d = DayanMxfp4Layout::make(ggml_nrows(dw), dw->ne[0]);
    DayanFfnConfig config = {};
    config.gate.m = config.down.m = 1;
    config.gate.k = gw->ne[0]; config.gate.n = gw->ne[1];
    config.down.k = dw->ne[0]; config.down.n = dw->ne[1];
    config.gate.a_stride = mx ? a.row_stride : input->nb[1];
    config.gate.a_scale_stride = a.scale_stride;
    config.gate.b_stride = mx ? g.row_stride : gw->nb[1];
    config.gate.b_scale_stride = g.scale_stride;
    config.down.b_stride = mx ? d.row_stride : dw->nb[1];
    config.down.b_scale_stride = d.scale_stride;
    config.gate.weight_type = config.down.weight_type = mx ? kDayanDataTypeMXFP4 : kDayanDataTypeBF16;
    config.down.output_type = output->type == GGML_TYPE_F32 ? kDayanDataTypeF32 : kDayanDataTypeBF16;
    config.experts = gw->ne[2]; config.experts_used = used;
    if (DayanFfnLayout::make(config).bytes > MATMUL_DAYAN_1C_URAM_BYTES) { return false; }
    using tensor_ref = ggml_backend_dayan_mega_parameters::tensor_ref;
    return parameters->add<mega::ffn_decode>(config, input, tensor_ref{input, mx ? a.scale_offset : 0},
        gw, tensor_ref{gw, mx ? g.scale_offset : 0}, uw, tensor_ref{uw, mx ? u.scale_offset : 0},
        dw, tensor_ref{dw, mx ? d.scale_offset : 0}, selected, weights, output);
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
    return parameters->add<mega::copy>(config, source, output);
}

static bool ggml_backend_dayan_mega_add_ssm_conv_split(ggml_backend_dayan_mega_parameters * parameters,
                                                        const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * conv = slot.nodes[0];
    ggml_tensor * concat = slot.nodes[2];
    ggml_tensor * copy = slot.nodes[3];
    ggml_tensor * output = slot.nodes[1];
    if (slot.kind != ggml_backend_dayan_slot_kind::ssm_conv_silu || concat == nullptr || copy == nullptr) {
        return false;
    }
    DayanSSMConvConfig config = {
        (uint32_t) concat->ne[0], (uint32_t) concat->ne[1], (uint32_t) output->ne[1],
        (uint32_t) concat->ne[2], (uint32_t) conv->src[1]->ne[0], kDayanSSMConvFlagSilu, 0, 0,
    };
    const float neg_log2e = -1.4426950408889634f;
    const float one = 1.0f;
    std::memcpy(&config.neg_log2e_bits, &neg_log2e, sizeof(neg_log2e));
    std::memcpy(&config.one_bits, &one, sizeof(one));
    return parameters->add<mega::ssm_conv_split>(config, concat->src[0], conv->src[1], output, concat->src[1], copy);
}

static bool ggml_backend_dayan_mega_add_l2_norm(ggml_backend_dayan_mega_parameters * parameters,
                                                 const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::l2_norm || output == nullptr || output->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanNormConfig config = ggml_backend_dayan_mega_norm_config(output);
    return parameters->add<mega::l2_norm>(config, output->src[0], output);
}

static bool ggml_backend_dayan_mega_add_binary(ggml_backend_dayan_mega_parameters * parameters,
                                                const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (output == nullptr || (slot.kind != ggml_backend_dayan_slot_kind::elementwise_add &&
                              slot.kind != ggml_backend_dayan_slot_kind::residual_add &&
                              slot.kind != ggml_backend_dayan_slot_kind::elementwise_mul &&
                              slot.kind != ggml_backend_dayan_slot_kind::elementwise_div) ||
        output->type != GGML_TYPE_F32) {
        return false;
    }
    // The full (output-shaped) operand is the left input; the other one may
    // broadcast. Swap operands only for commutative operations.
    const ggml_tensor * left = output->src[0];
    const ggml_tensor * right = output->src[1];
    if (!ggml_are_same_shape(output, left) && slot.kind != ggml_backend_dayan_slot_kind::elementwise_div) {
        const ggml_tensor * swap = left;
        left = right;
        right = swap;
    }
    if (!ggml_are_same_shape(output, left)) {
        return false;
    }
    if (left->nb[0] != sizeof(float) || right->nb[0] != sizeof(float)) { return false; }
    DayanBinaryConfig config = {};
    for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        if (!ggml_is_contiguous(left)) { config.left_nb[dimension - 1] = left->nb[dimension]; }
        if (!ggml_is_contiguous(right)) { config.right_nb[dimension - 1] = right->nb[dimension]; }
    }
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
    switch (slot.kind) {
        case ggml_backend_dayan_slot_kind::elementwise_mul: return parameters->add<mega::multiply>(config, left, right, output);
        case ggml_backend_dayan_slot_kind::elementwise_div: return parameters->add<mega::divide>(config, left, right, output);
        default: return parameters->add<mega::add>(config, left, right, output);
    }
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
    switch (slot.kind) {
        case ggml_backend_dayan_slot_kind::softplus: return parameters->add<mega::softplus>(config, output->src[0], output);
        case ggml_backend_dayan_slot_kind::sigmoid: return parameters->add<mega::sigmoid>(config, output->src[0], output);
        default: return parameters->add<mega::silu>(config, output->src[0], output);
    }
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
    return parameters->add<mega::gated_delta_net>(config, q, k, v, g, beta, state, output);
}

static bool ggml_backend_dayan_mega_add_gdn_z_gate(ggml_backend_dayan_mega_parameters * parameters,
                                                    const std::vector<ggml_backend_dayan_graph_slot> & slots,
                                                    size_t first) {
    if (first > slots.size() || slots.size() - first < 6) { return false; }
    const auto & gdn_slot = slots[first];
    const auto & copy_slot = slots[first + 1];
    const auto & norm_slot = slots[first + 2];
    const auto & z_slot = slots[first + 3];
    const auto & silu_slot = slots[first + 4];
    const auto & gate_slot = slots[first + 5];
    ggml_tensor * gdn_output = gdn_slot.nodes[0];
    ggml_tensor * state_copy = copy_slot.nodes[0];
    ggml_tensor * rms = norm_slot.nodes[1];
    ggml_tensor * scaled = norm_slot.nodes[2];
    ggml_tensor * lowp = norm_slot.nodes[3];
    const ggml_tensor * state_source = state_copy != nullptr ? state_copy->src[0] : nullptr;
    ggml_tensor * z = z_slot.nodes[0];
    ggml_tensor * z_silu = silu_slot.nodes[0];
    ggml_tensor * gate = gate_slot.nodes[0];
    if (gdn_slot.kind != ggml_backend_dayan_slot_kind::gated_delta_net ||
        copy_slot.kind != ggml_backend_dayan_slot_kind::copy ||
        norm_slot.kind != ggml_backend_dayan_slot_kind::norm_pipeline || norm_slot.nodes[0] != nullptr ||
        z_slot.kind != ggml_backend_dayan_slot_kind::mul_mat || silu_slot.kind != ggml_backend_dayan_slot_kind::silu ||
        gate_slot.kind != ggml_backend_dayan_slot_kind::elementwise_mul || gdn_output == nullptr || state_copy == nullptr ||
        rms == nullptr || scaled == nullptr || z == nullptr || z_silu == nullptr || gate == nullptr ||
        state_source == nullptr || z->src[0] == nullptr || z->src[1] == nullptr || z_silu->src[0] == nullptr ||
        (z_silu->src[0] != z && (z_silu->src[0]->view_src != z || z_silu->src[0]->view_offs != 0)) ||
        (gate->src[0] != scaled && gate->src[1] != scaled) ||
        (gate->src[0] != z_silu && gate->src[1] != z_silu) ||
        ggml_get_op_params_i32(gdn_output, 0) != 1 ||
        (uint64_t) ggml_nelements(state_source) != (uint64_t) ggml_nelements(state_copy) ||
        state_source->view_src != gdn_output || z->src[1]->type != GGML_TYPE_BF16 ||
        z->src[0]->type != GGML_TYPE_BF16 || z->type != GGML_TYPE_F32 || z_silu->type != GGML_TYPE_F32 ||
        gate->type != GGML_TYPE_F32 || !ggml_is_contiguous(z_silu) || !ggml_is_contiguous(gate) ||
        !ggml_are_same_shape(z, z_silu) || !ggml_are_same_shape(gate, scaled) ||
        !ggml_are_same_shape(gate, z_silu) || (z->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }

    const ggml_tensor * q = gdn_output->src[0];
    const ggml_tensor * k = gdn_output->src[1];
    const ggml_tensor * v = gdn_output->src[2];
    const ggml_tensor * g = gdn_output->src[3];
    const ggml_tensor * beta = gdn_output->src[4];
    const ggml_tensor * state = gdn_output->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr || state == nullptr ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        v->ne[2] != 1 || v->ne[3] != 1 || gdn_output->ne[2] != 1 || gdn_output->ne[3] != 1 || rms->src[0] == nullptr ||
        rms->src[0]->view_src != gdn_output || rms->src[0]->view_offs != 0 ||
        !ggml_backend_dayan_supports_rms_norm_scale(scaled) || z->src[1]->ne[0] != z->src[0]->ne[0] ||
        z->src[1]->ne[1] != 1 || z->src[1]->ne[2] != 1 || z->src[1]->ne[3] != 1 ||
        z->ne[0] != z->src[0]->ne[1] || z->ne[1] != z->src[1]->ne[1] || z->ne[2] != 1 || z->ne[3] != 1) {
        return false;
    }

    DayanGdnZGateConfig config = {};
    config.gdn.head_dim = (uint32_t) v->ne[0];
    config.gdn.heads = (uint32_t) v->ne[1];
    config.gdn.tokens = (uint32_t) v->ne[2];
    config.gdn.sequences = (uint32_t) v->ne[3];
    if (config.gdn.tokens != 1 || config.gdn.sequences != 1) { return false; }
    config.gdn.snapshots = (uint32_t) ggml_get_op_params_i32(gdn_output, 0);
    config.gdn.gate_width = (uint32_t) g->ne[0];
    config.gdn.q_heads = (uint32_t) q->ne[1];
    config.gdn.k_heads = (uint32_t) k->ne[1];
    config.gdn.q_sequences = (uint32_t) q->ne[3];
    config.gdn.k_sequences = (uint32_t) k->ne[3];
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.gdn.q_nb[dimension] = q->nb[dimension];
        config.gdn.k_nb[dimension] = k->nb[dimension];
        config.gdn.v_nb[dimension] = v->nb[dimension];
        config.gdn.g_nb[dimension] = g->nb[dimension];
        config.gdn.beta_nb[dimension] = beta->nb[dimension];
    }
    const float gdn_scale = 1.0f / std::sqrt((float) config.gdn.head_dim);
    std::memcpy(&config.gdn.scale_bits, &gdn_scale, sizeof(gdn_scale));

    if (state_source->view_offs != (size_t) config.gdn.tokens * config.gdn.sequences * config.gdn.heads *
                                      config.gdn.head_dim * sizeof(float) ||
        state_copy->type != GGML_TYPE_F32 || !ggml_is_contiguous(state_copy) || state_source->type != GGML_TYPE_F32 ||
        rms->src[0]->type != GGML_TYPE_F32 || scaled->type != GGML_TYPE_F32 ||
        (lowp != nullptr && lowp->type != GGML_TYPE_BF16)) {
        return false;
    }
    config.state_copy.elements = (uint64_t) ggml_nelements(state_copy);
    config.state_copy.source_type = kDayanDataTypeF32;
    config.state_copy.destination_type = kDayanDataTypeF32;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.state_copy.source_ne[dimension] = (uint32_t) state_source->ne[dimension];
        config.state_copy.source_nb[dimension] = state_source->nb[dimension];
    }

    const ggml_tensor * norm_scale = ggml_backend_dayan_mega_norm_scale(scaled, rms);
    if (norm_scale == nullptr || norm_scale->type != GGML_TYPE_F32) {
        return false;
    }
    config.norm_pipeline.norm = ggml_backend_dayan_mega_norm_config(rms);
    config.norm_pipeline.flags = 0;
    if (norm_slot.output_mask & kDayanNormSlotOutputRms) { config.norm_pipeline.flags |= kDayanNormWriteRms; }
    if (norm_slot.output_mask & kDayanNormSlotOutputScaled) { config.norm_pipeline.flags |= kDayanNormWriteScaled; }
    if (norm_slot.output_mask & kDayanNormSlotOutputLowp) {
        if (lowp == nullptr || !ggml_backend_dayan_mega_type_code(lowp->type, &config.norm_pipeline.output_type)) {
            return false;
        }
        config.norm_pipeline.flags |= kDayanNormWriteLowp;
    }
    if ((config.norm_pipeline.flags & kDayanNormWriteScaled) == 0 ||
        ((config.norm_pipeline.flags & kDayanNormWriteLowp) != 0 && config.norm_pipeline.output_type != kDayanDataTypeBF16)) {
        return false;
    }

    config.z_bf16 = ggml_backend_dayan_mega_matmul_config(z->src[0], z->src[1], z);
    if (config.z_bf16.m > 512) { return false; }
    config.z_bf16.uram_offset = 256U * 1024U;
    config.z_silu.elements = (uint64_t) ggml_nelements(z_silu);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        config.gate_mul.ne[dimension] = (uint32_t) gate->ne[dimension];
        config.gate_mul.right_ne[dimension] = (uint32_t) z_silu->ne[dimension];
    }
    config.gate_mul.elements = (uint64_t) ggml_nelements(gate);

    return parameters->add<mega::gated_delta_net_z_bf16>(config, q, k, v, g, beta, state, gdn_output,
        state_source, state_copy, norm_scale, rms, scaled, lowp != nullptr ? lowp : scaled,
        z->src[1], z->src[0], z, z_silu, gate);
}

static bool ggml_backend_dayan_mega_add_rms_norm(ggml_backend_dayan_mega_parameters * parameters,
                                                  const ggml_backend_dayan_graph_slot & slot) {
    ggml_tensor * output = slot.nodes[0];
    if (slot.kind != ggml_backend_dayan_slot_kind::rms_norm || output == nullptr || output->type != GGML_TYPE_F32) {
        return false;
    }
    const DayanNormConfig config = ggml_backend_dayan_mega_norm_config(output);
    return parameters->add<mega::rms_norm>(config, output->src[0], output);
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
    if (factors != nullptr) {
        return parameters->add<mega::rope>(config, input, positions, factors, output);
    }
    const std::vector<float> ones((size_t) n_dims / 2, 1.0f);
    const ggml_backend_dayan_mega_parameters::data_ref factor_data = {
        ones.data(), ones.size() * sizeof(float), alignof(float),
    };
    return parameters->add<mega::rope>(config, input, positions, factor_data, output);
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
    return parameters->add<mega::flash_attention_fp16>(config, q, k, v, output);
}

static bool ggml_backend_dayan_mega_add_bf16_gated_projection(
    ggml_backend_dayan_mega_parameters * parameters,
    const ggml_backend_dayan_graph_slot & slot,
    npu_buffer_t *                       scratch,
    uint64_t                             scratch_bytes) {
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
        output->type != GGML_TYPE_F32) {
        return false;
    }
    const uint64_t tokens   = attention->ne[1];
    const uint64_t heads    = gate_view->ne[1];
    const uint64_t head_dim = gate_view->ne[0];
    const uint64_t width    = heads * head_dim;
    if (tokens > UINT32_MAX || heads > UINT32_MAX || head_dim > UINT32_MAX ||
        width > SIZE_MAX / sizeof(uint16_t) || tokens > SIZE_MAX / (width * sizeof(uint16_t)) ||
        scratch_bytes < (size_t) tokens * width * sizeof(uint16_t)) {
        return false;
    }
    const DayanGatedProjectionConfig config = {
        ggml_backend_dayan_mega_matmul_config(weight, qg->src[1], output),
        (uint32_t) tokens,
        (uint32_t) heads,
        (uint32_t) head_dim,
    };
    return parameters->add<mega::gated_output_projection_bf16>(config, attention, qg, weight, output, scratch);
}

static bool ggml_backend_dayan_mega_add_tf32(ggml_backend_dayan_mega_parameters * parameters,
                                             const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * output = slot.nodes[0];
    if (output == nullptr || output->src[0] == nullptr || output->src[1] == nullptr ||
        output->src[0]->type != GGML_TYPE_F32 || output->src[1]->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return false;
    }
    DayanMatmulConfig config = ggml_backend_dayan_mega_matmul_config(output->src[0], output->src[1], output);
    config.a_stride = config.k * sizeof(float);
    return parameters->add<mega::matmul_tf32>(config, output->src[1], output->src[0], output);
}

static bool ggml_backend_dayan_mega_add_moe_sort(ggml_backend_dayan_mega_parameters * parameters,
                                                        const ggml_backend_dayan_graph_slot & slot,
                                                        const ggml_tensor *                  selected,
                                                        npu_buffer_t *                       workspace,
                                                        uint64_t                             workspace_bytes) {
    const ggml_tensor * output = slot.nodes[0];
    if (output == nullptr || output->src[0] == nullptr || selected == nullptr || output->op != GGML_OP_ARGSORT ||
        output->src[0]->type != GGML_TYPE_F32 || output->type != GGML_TYPE_I32 || selected->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(output->src[0]) || !ggml_is_contiguous(output) || selected->nb[0] != sizeof(int32_t) ||
        selected->ne[0] <= 0 || selected->ne[0] > slot.sort_k || selected->view_src != output || selected->view_offs != 0 ||
        selected->ne[1] != output->ne[1] || selected->nb[1] != output->nb[1] ||
        ggml_get_op_params_i32(output, 0) != GGML_SORT_ORDER_DESC) {
        return false;
    }
    DayanMoeSortConfig config = {};
    config.sort = {(uint32_t) output->src[0]->ne[0], (uint32_t) ggml_nrows(output->src[0]), slot.sort_k,
                   kDayanSortOrderDesc, (uint32_t) output->ne[0]};
    config.experts_used = (uint32_t) selected->ne[0];
    config.workspace_bytes = workspace_bytes;
    return parameters->add<mega::moe_sort>(config, output->src[0], output, workspace);
}

static bool ggml_backend_dayan_mega_add_routing(ggml_backend_dayan_mega_parameters * parameters,
                                                const ggml_backend_dayan_graph_slot & slot) {
    const ggml_tensor * output = slot.nodes[0];
    if (output == nullptr || output->src[0] == nullptr) { return false; }
    switch (slot.kind) {
        case ggml_backend_dayan_slot_kind::softmax: {
            if (output->src[1] != nullptr || output->src[2] != nullptr) { return false; }
            DayanSoftmaxConfig config = {(uint32_t) output->ne[0], (uint32_t) output->ne[1],
                (uint32_t) output->ne[2], (uint32_t) output->ne[3], 1, 1, 1, kDayanDataTypeF32, 0, 0, 0, 0};
            std::memcpy(&config.scale_bits, output->op_params, sizeof(config.scale_bits));
            std::memcpy(&config.max_bias_bits, (const char *) output->op_params + sizeof(float), sizeof(config.max_bias_bits));
            return parameters->add<mega::softmax>(config, output->src[0], output);
        }
        case ggml_backend_dayan_slot_kind::sum_rows: {
            const DayanReduceConfig config = {(uint32_t) output->src[0]->ne[0], (uint32_t) ggml_nrows(output), 0x3f800000U, 0};
            return parameters->add<mega::sum_rows>(config, output->src[0], output);
        }
        case ggml_backend_dayan_slot_kind::clamp: {
            DayanClampConfig config = {(uint64_t) ggml_nelements(output), 0, 0};
            std::memcpy(&config.min_bits, output->op_params, sizeof(config.min_bits));
            std::memcpy(&config.max_bits, (const char *) output->op_params + sizeof(float), sizeof(config.max_bits));
            return parameters->add<mega::clamp>(config, output->src[0], output);
        }
        default: return false;
    }
}

enum class ggml_backend_dayan_mega_slot_mode { required, optional, reuse, borrow, fused };

struct ggml_backend_dayan_mega_context {
    const std::vector<ggml_backend_dayan_graph_slot> & slots;
    std::vector<size_t> indices;
    npu_buffer_t * workspace = nullptr;
    uint64_t workspace_bytes = 0;

    bool missing(unsigned step) const { return indices[step] == SIZE_MAX; }
};

struct ggml_backend_dayan_mega_step {
    ggml_backend_dayan_slot_kind kind;
    ggml_backend_dayan_mega_slot_mode mode;
    uint16_t argument_count;
    bool (*pack)(ggml_backend_dayan_mega_parameters *, const ggml_backend_dayan_mega_context &);
};

namespace ggml_dayan_mega_detail {

#define pack(Method, ...) ggml_backend_dayan_mega_add_##Method(parameters, __VA_ARGS__)
#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) namespace Name { static const ggml_backend_dayan_mega_step steps[] = {
#define MEGA_STEP(Name, Kind, Function, Barriers, Mode, ...) \
    { ggml_backend_dayan_slot_kind::Kind, ggml_backend_dayan_mega_slot_mode::Mode, Signature<decltype(&ggml_dayan_mega_detail::Function)>::count, \
      [](ggml_backend_dayan_mega_parameters * parameters, const ggml_backend_dayan_mega_context & ctx) { \
          const auto at = [&ctx](Step step) -> const ggml_backend_dayan_graph_slot & { return ctx.slots[ctx.indices[step]]; }; \
          parameters->expect<&ggml_dayan_mega_detail::Function>(); \
          return (__VA_ARGS__); \
      } },
#define MEGA_GRAPH_END() }; }
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END
#undef pack

} // namespace ggml_dayan_mega_detail

struct ggml_backend_dayan_mega_graph {
    uint16_t id;
    bool moe;
    const ggml_backend_dayan_mega_step * steps;
    size_t step_count;
    uint16_t parameter_count;
};

static const ggml_backend_dayan_mega_graph ggml_backend_dayan_mega_graphs[] = {
#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) { Id, Moe, mega::Name::steps, mega::Name::step_count, mega::Name::parameter_count },
#define MEGA_STEP(...)
#define MEGA_GRAPH_END()
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END
};

static bool ggml_backend_dayan_mega_match(const ggml_backend_dayan_mega_graph & graph,
                                         ggml_backend_dayan_mega_context & ctx, size_t begin, size_t * end) {
    ctx.indices.assign(graph.step_count, SIZE_MAX);
    size_t cursor = begin;
    for (size_t index = 0; index < graph.step_count; ++index) {
        const auto & step = graph.steps[index];
        const bool reuse = step.mode == ggml_backend_dayan_mega_slot_mode::reuse;
        if (reuse && cursor == begin) {
            return false;
        }
        const size_t position = reuse ? cursor - 1 : cursor;
        if (position >= ctx.slots.size()) {
            return false;
        }
        const auto actual = ctx.slots[position].kind;
        const bool add = (step.kind == ggml_backend_dayan_slot_kind::elementwise_add ||
                          step.kind == ggml_backend_dayan_slot_kind::residual_add) &&
                         (actual == ggml_backend_dayan_slot_kind::elementwise_add ||
                          actual == ggml_backend_dayan_slot_kind::residual_add);
        if (actual != step.kind && !add) {
            if (step.mode == ggml_backend_dayan_mega_slot_mode::optional) {
                continue;
            }
            return false;
        }
        ctx.indices[index] = position;
        if (step.mode == ggml_backend_dayan_mega_slot_mode::fused) {
            cursor += 6;
        } else {
            cursor += !reuse && step.mode != ggml_backend_dayan_mega_slot_mode::borrow;
        }
    }
    *end = cursor;
    return true;
}

static bool ggml_backend_dayan_mega_workspace(const ggml_backend_dayan_mega_runtime & runtime,
                                             const ggml_backend_dayan_mega_graph & graph,
                                             ggml_backend_dayan_mega_context & ctx) {
    ctx.workspace_bytes = graph.moe ? 64 : 0;
    for (size_t index = 0; index < graph.step_count; ++index) {
        const auto kind = graph.steps[index].kind;
        if (kind == ggml_backend_dayan_slot_kind::gated_output_projection) {
            const auto & slot = ctx.slots[ctx.indices[index]];
            const auto * projection = slot.nodes[2];
            if (projection == nullptr || projection->src[1] == nullptr) { return false; }
            // Graph tensor storage may alias the projection output after attention.
            const uint64_t elements = ggml_nelements(projection->src[1]);
            if (elements > SIZE_MAX / sizeof(uint16_t)) { return false; }
            ctx.workspace_bytes = std::max(ctx.workspace_bytes, elements * sizeof(uint16_t));
            continue;
        }
        const bool glu = kind == ggml_backend_dayan_slot_kind::mul_mat_id_swiglu;
        if (!glu && kind != ggml_backend_dayan_slot_kind::mul_mat_id) {
            continue;
        }
        const auto & slot = ctx.slots[ctx.indices[index]];
        const ggml_tensor * node = slot.nodes[0];
        const ggml_tensor * output = glu ? slot.nodes[3] : node;
        if (node == nullptr || node->src[0] == nullptr || node->src[2] == nullptr || output == nullptr ||
            node->src[2]->ne[0] != 8 ||
            (glu && (slot.nodes[1] == nullptr || slot.nodes[1]->src[2] != node->src[2]))) {
            return false;
        }
        const uint64_t tokens = node->src[2]->ne[1];
        const uint64_t used = node->src[2]->ne[0];
        if (tokens > 1) {
            const auto dispatch = DayanMoeDispatchLayout::make(tokens, node->src[0]->ne[2], used);
            const auto input_layout = DayanMxfp4Layout::make(tokens * used, node->src[0]->ne[0]);
            const auto output_layout = DayanMxfp4Layout::make(tokens * used, output->ne[0]);
            const uint64_t output_bytes = output->type == GGML_TYPE_MXFP4 ? output_layout.bytes :
                                          tokens * used * output->ne[0] * sizeof(float);
            const uint64_t bytes = dispatch.bytes + ((input_layout.bytes + 63U) & ~63ULL) + ((output_bytes + 63U) & ~63ULL);
            ctx.workspace_bytes = std::max(ctx.workspace_bytes, bytes);
        }
    }
    return ctx.workspace_bytes == 0 || (runtime.prepare_workspace != nullptr &&
           runtime.prepare_workspace(runtime.context, ctx.workspace_bytes, &ctx.workspace) == 0);
}

ggml_backend_dayan_mega_result ggml_backend_dayan_try_qwen35_mega(
    const ggml_backend_dayan_mega_runtime & runtime,
    const std::vector<ggml_backend_dayan_graph_slot> & slots,
    size_t begin) {
    if (!runtime.enabled || runtime.resolve == nullptr || runtime.launch == nullptr) {
        return { 0, begin, false };
    }
    for (const auto & graph : ggml_backend_dayan_mega_graphs) {
        ggml_backend_dayan_mega_context ctx = { slots, {} };
        size_t end = begin;
        if (!ggml_backend_dayan_mega_match(graph, ctx, begin, &end) ||
            !ggml_backend_dayan_mega_workspace(runtime, graph, ctx)) {
            continue;
        }
        ggml_backend_dayan_mega_parameters parameters(runtime, graph.id, graph.parameter_count);
        bool packed = true;
        for (size_t index = 0; index < graph.step_count; ++index) {
            const auto & step = graph.steps[index];
            const uint16_t first = parameters.next();
            if (!step.pack(&parameters, ctx) || parameters.next() - first != step.argument_count) {
                packed = false;
                break;
            }
        }
        if (!packed || !parameters.finish()) {
            continue;
        }
        npu_buffer_t * roots[kDayanMegaRootCount] = {};
        for (size_t index = 0; index < kDayanMegaRootCount; ++index) {
            roots[index] = parameters.root(index);
        }
        const int status = runtime.launch(runtime.context, graph.id, parameters.data().data(), parameters.data().size(), roots);
        return { status, end, true };
    }
    return { 0, begin, false };
}
