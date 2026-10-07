#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-dayan.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <vector>

struct layer_operator_compare {
    using json = nlohmann::json;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu{ggml_backend_cpu_init(), ggml_backend_free};
    json nodes = json::array();
    json pending;
    std::vector<float> expected;
    int step = 0;
    size_t slots = 0;
    std::unordered_set<const ggml_tensor *> computed;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> copies;
    std::unordered_map<const ggml_tensor *, std::vector<float>> references;
    std::unordered_map<const ggml_tensor *, std::vector<int64_t>> integer_references;
    ggml_backend_dayan_set_slot_filter_callback_t set_filter = nullptr;
    ggml_backend_dayan_set_slot_trace_callback_t set_trace = nullptr;
    size_t failures = 0;
    double atol = 1e-4;
    double rtol = 1e-3;

    layer_operator_compare() {
        if (!cpu) { throw std::runtime_error("cannot initialize CPU reference backend"); }
        ggml_backend_cpu_set_n_threads(cpu.get(), 8);
        ggml_backend_cpu_set_use_ref(cpu.get(), true);
    }

    ~layer_operator_compare() {
        if (set_filter) { set_filter(nullptr, nullptr); }
        if (set_trace) { set_trace(nullptr, nullptr); }
    }

    void attach(ggml_backend_dev_t device) {
        const auto reg = ggml_backend_dev_backend_reg(device);
        set_filter = (ggml_backend_dayan_set_slot_filter_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dayan_set_slot_filter_callback");
        set_trace = (ggml_backend_dayan_set_slot_trace_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dayan_set_slot_trace_callback");
        if (!set_filter || !set_trace) { throw std::runtime_error("Dayan slot callbacks are unavailable"); }
        set_filter(before, this);
        set_trace(after, this);
    }

    static bool view(const ggml_tensor * t) {
        return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
    }

    bool internal(const ggml_tensor * t) const {
        if (computed.count(t)) { return true; }
        return view(t) && t->view_src && internal(t->view_src);
    }

    ggml_tensor * clone(ggml_context * ctx, const ggml_tensor * source, bool half = false) {
        if (copies.size() >= 240) { throw std::runtime_error("reference slot has too many tensors"); }
        const auto found = copies.find(source);
        if (found != copies.end()) { return found->second; }
        ggml_tensor * target;
        if (internal(source) && view(source)) {
            ggml_tensor * base = clone(ctx, source->view_src);
            if (ggml_is_quantized(source->type)) { throw std::runtime_error("quantized internal views are unsupported"); }
            const size_t bytes = ggml_type_size(source->type), target_bytes = ggml_type_size(base->type);
            target = ggml_view_4d(ctx, base, source->ne[0], source->ne[1], source->ne[2], source->ne[3],
                                 source->nb[1] / bytes * target_bytes, source->nb[2] / bytes * target_bytes, source->nb[3] / bytes * target_bytes, source->view_offs / bytes * target_bytes);
            target->nb[0] = source->nb[0] / bytes * target_bytes;
        } else {
            target = ggml_new_tensor(ctx, integer(source->type) ? source->type : half ? GGML_TYPE_F16 : GGML_TYPE_F32, 4, source->ne);
            if (internal(source)) {
                target->op = source->op;
                std::memcpy(target->op_params, source->op_params, sizeof(target->op_params));
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (source->src[i]) { target->src[i] = clone(ctx, source->src[i], source->op == GGML_OP_FLASH_ATTN_EXT && i == 3); }
                }
            }
        }
        ggml_set_name(target, source->name);
        copies[source] = target;
        return target;
    }

    static bool integer(ggml_type type) { return type == GGML_TYPE_I32 || type == GGML_TYPE_I64; }

    template <typename T = float> static std::vector<T> values(const ggml_tensor * tensor) {
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        std::vector<T> result(ggml_nelements(tensor));
        const auto * traits = ggml_get_type_traits(tensor->type);
        size_t index = 0;
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    const auto * row = bytes.data() + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
                    if (ggml_is_quantized(tensor->type)) {
                        if (!traits->to_float || tensor->nb[0] != ggml_type_size(tensor->type)) {
                            throw std::runtime_error("unsupported quantized reference layout");
                        }
                        if constexpr (std::is_same<T, float>::value) { traits->to_float(row, result.data() + index, tensor->ne[0]); }
                        else { throw std::runtime_error("invalid integer reference type"); }
                        index += tensor->ne[0];
                        continue;
                    }
                    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                        const void * element = row + i0 * tensor->nb[0];
                        T value;
                        switch (tensor->type) {
                            case GGML_TYPE_F32: { float x; std::memcpy(&x, element, 4); value = x; break; }
                            case GGML_TYPE_BF16: { ggml_bf16_t x; std::memcpy(&x, element, 2); value = ggml_bf16_to_fp32(x); break; }
                            case GGML_TYPE_F16: { ggml_fp16_t x; std::memcpy(&x, element, 2); value = ggml_fp16_to_fp32(x); break; }
                            case GGML_TYPE_I32: { int32_t x; std::memcpy(&x, element, 4); value = x; break; }
                            case GGML_TYPE_I64: { int64_t x; std::memcpy(&x, element, 8); value = x; break; }
                            default: throw std::runtime_error("unsupported reference tensor type");
                        }
                        result[index++] = value;
                    }
                }
            }
        }
        return result;
    }

    static void snapshot(const ggml_tensor * source, ggml_tensor * target) {
        if (!integer(source->type)) {
            auto data = values(source);
            if (target->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(data.size());
                ggml_fp32_to_fp16_row(data.data(), half.data(), data.size());
                ggml_backend_tensor_set(target, half.data(), 0, half.size() * sizeof(half[0]));
            } else {
                ggml_backend_tensor_set(target, data.data(), 0, data.size() * sizeof(float));
            }
            return;
        }
        std::vector<uint8_t> bytes(ggml_nbytes(source));
        ggml_backend_tensor_get(source, bytes.data(), 0, bytes.size());
        auto * output = (uint8_t *) target->data;
        const size_t element = ggml_type_size(source->type);
        for (int64_t i3 = 0; i3 < source->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < source->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < source->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < source->ne[0]; ++i0) {
                        std::memcpy(output, bytes.data() + i0 * source->nb[0] + i1 * source->nb[1] + i2 * source->nb[2] + i3 * source->nb[3], element);
                        output += element;
                    }
                }
            }
        }
    }

    void reference(const ggml_backend_dayan_slot_trace_event * event) {
        computed.clear(); copies.clear(); references.clear(); integer_references.clear();
        for (size_t i = 0; i < GGML_DAYAN_SLOT_TRACE_MAX_NODES; ++i) {
            if (event->nodes[i] && !(std::strcmp(event->kind, "gated_output_projection") == 0 && i == 3)) {
                computed.insert(event->nodes[i]);
            }
        }
        for (size_t i = 0; i < event->output_count; ++i) { computed.insert(event->outputs[i]); }
        ggml_init_params params{ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        if (!ctx) { throw std::runtime_error("cannot create CPU reference graph"); }
        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        for (size_t i = 0; i < event->output_count; ++i) {
            const auto * tensor = event->outputs[i];
            if (ggml_is_empty(tensor)) { references[tensor] = {}; continue; }
            ggml_build_forward_expand(graph, clone(ctx.get(), tensor));
        }
        if (copies.empty()) { return; }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()), ggml_backend_buffer_free);
        if (!buffer) { throw std::runtime_error("cannot allocate CPU reference tensors"); }
        pending["inputs"] = json::array();
        for (const auto & copy : copies) {
            if (!internal(copy.first)) {
                const auto * t = copy.first;
                pending["inputs"].push_back({{"name", t->name}, {"type", ggml_type_name(t->type)},
                                              {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}}});
            }
            if (!internal(copy.first) || copy.first->op == GGML_OP_SET_ROWS) { snapshot(copy.first, copy.second); }
            if (!ggml_backend_supports_op(cpu.get(), copy.second)) { throw std::runtime_error("CPU does not support reference operation"); }
        }
        if (ggml_backend_graph_compute(cpu.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CPU reference operation failed");
        }
        for (size_t i = 0; i < event->output_count; ++i) {
            const auto * tensor = event->outputs[i];
            if (ggml_is_empty(tensor)) { continue; }
            if (integer(tensor->type)) { integer_references[tensor] = values<int64_t>(copies.at(tensor)); }
            expected = values(copies.at(tensor));
            round_output(tensor);
            references[tensor] = std::move(expected);
        }
    }

    void round_output(const ggml_tensor * tensor) {
        if (tensor->type == GGML_TYPE_BF16) {
            for (auto & x : expected) { x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)); }
        } else if (tensor->type == GGML_TYPE_F16) {
            for (auto & x : expected) { x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
        } else if (ggml_is_quantized(tensor->type)) {
            std::vector<uint8_t> bytes(ggml_row_size(tensor->type, expected.size()));
            ggml_quantize_chunk(tensor->type, expected.data(), bytes.data(), 0, ggml_nrows(tensor), tensor->ne[0], nullptr);
            ggml_get_type_traits(tensor->type)->to_float(bytes.data(), expected.data(), expected.size());
        }
    }

    void compare(const ggml_tensor * tensor) {
        const auto actual = values(tensor);
        const auto integers = integer(tensor->type) ? values<int64_t>(tensor) : std::vector<int64_t>{};
        if (actual.size() != expected.size()) { throw std::runtime_error("reference output shape changed"); }
        size_t mismatches = 0, nonfinite = 0;
        double max_abs = 0, max_rel = 0, error2 = 0, reference2 = 0;
        json first = nullptr;
        const double absolute = integer(tensor->type) ? 0 : atol;
        const double relative = integer(tensor->type) ? 0 : rtol;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (integer(tensor->type)) {
                const int64_t reference = integer_references.at(tensor).at(i);
                if (integers[i] != reference) {
                    ++mismatches;
                    const double error = std::abs((long double) integers[i] - reference);
                    max_abs = std::max(max_abs, error);
                    max_rel = std::max(max_rel, error / std::max(std::abs(double(reference)), 1.0));
                    error2 += error * error;
                    if (first.is_null()) { first = {{"index", i}, {"actual", integers[i]}, {"expected", reference}}; }
                }
                reference2 += double(reference) * reference;
                continue;
            }
            if (actual[i] == expected[i]) { reference2 += std::isfinite(expected[i]) ? double(expected[i]) * expected[i] : 0; continue; }
            const bool finite = std::isfinite(actual[i]) && std::isfinite(expected[i]);
            const double error = finite ? std::abs(double(actual[i]) - expected[i]) : 0;
            if (!finite) { ++nonfinite; }
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, error / std::max(std::abs(double(expected[i])), 1e-12));
            error2 += error * error;
            if (finite) { reference2 += double(expected[i]) * expected[i]; }
            if (!finite || error > absolute + relative * std::abs(expected[i])) {
                ++mismatches;
                if (first.is_null()) { first = {{"index", i}, {"actual", actual[i]}, {"expected", expected[i]}}; }
            }
        }
        pending.update({{"status", mismatches ? "fail" : "pass"}, {"checked_elements", actual.size()},
                        {"mismatched_elements", mismatches}, {"nonfinite_elements", nonfinite},
                        {"max_abs_error", max_abs}, {"max_rel_error", max_rel},
                        {"relative_l2_error", std::sqrt(error2 / std::max(reference2, 1e-30))}, {"first_mismatch", first}});
        failures += mismatches != 0;
        std::fprintf(stderr, "CPU op step=%d %s %s: %s max_abs=%.8g mismatches=%zu/%zu\n", step,
                     ggml_op_name(tensor->op), tensor->name, mismatches ? "FAIL" : "PASS", max_abs, mismatches, actual.size());
        nodes.push_back(pending);
    }

    static bool before(const ggml_backend_dayan_slot_trace_event * event, void * opaque) {
        auto & self = *(layer_operator_compare *) opaque;
        self.pending = {{"step", self.step}, {"slot", event->slot_index}, {"kind", event->kind}};
        json covered = json::array();
        for (const auto * t : event->nodes) {
            if (t) { covered.push_back({{"name", t->name}, {"op", ggml_op_name(t->op)}}); }
        }
        self.pending["fused_nodes"] = covered;
        try {
            if (event->version != 4 || !event->output_count || event->output_count > GGML_DAYAN_SLOT_TRACE_MAX_OUTPUTS) {
                throw std::runtime_error("unsupported Dayan slot event");
            }
            self.reference(event);
        }
        catch (const std::exception & error) {
            self.pending.update({{"status", "error"}, {"error", error.what()}});
            self.nodes.push_back(self.pending);
            ++self.failures;
            throw;
        }
        return true;
    }

    static void after(const ggml_backend_dayan_slot_trace_event * event, void * opaque) {
        auto & self = *(layer_operator_compare *) opaque;
        if (!event->executed || event->status != 0) { throw std::runtime_error("Dayan slot execution failed"); }
        ++self.slots;
        for (size_t i = 0; i < event->output_count; ++i) {
            const auto * t = event->outputs[i];
            self.pending.update({{"name", t->name}, {"op", ggml_op_name(t->op)}, {"type", ggml_type_name(t->type)},
                                 {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}}});
            self.expected = std::move(self.references.at(t));
            self.compare(t);
        }
    }

    json report() const {
        return {{"reference", "CPU F32 with actual stored inputs and output rounding"},
                {"status", failures || nodes.empty() ? "fail" : "pass"}, {"checked_outputs", nodes.size()}, {"checked_nodes", nodes.size()}, {"checked_slots", slots}, {"granularity", "Dayan slots with fused operators preserved"}, {"failed_nodes", failures},
                {"atol", atol}, {"rtol", rtol}, {"nodes", nodes}};
    }
};
