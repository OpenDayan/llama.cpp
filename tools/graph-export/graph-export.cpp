#include "../src/llama-ext.h"
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml-dayan.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

static json tensor_dimensions(const int64_t * values) {
    json result = json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.push_back(values[i]);
    }
    return result;
}

static json tensor_strides(const size_t * values) {
    json result = json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.push_back(values[i]);
    }
    return result;
}

static json tensor_flags(const ggml_tensor * tensor) {
    json result = json::array();
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        result.push_back("input");
    }
    if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        result.push_back("output");
    }
    if (tensor->flags & GGML_TENSOR_FLAG_PARAM) {
        result.push_back("parameter");
    }
    if (tensor->flags & GGML_TENSOR_FLAG_LOSS) {
        result.push_back("loss");
    }
    if (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) {
        result.push_back("compute");
    }
    return result;
}

static const char * buffer_usage_name(enum ggml_backend_buffer_usage usage) {
    switch (usage) {
        case GGML_BACKEND_BUFFER_USAGE_ANY:
            return "any";
        case GGML_BACKEND_BUFFER_USAGE_WEIGHTS:
            return "weights";
        case GGML_BACKEND_BUFFER_USAGE_COMPUTE:
            return "compute";
    }
    return "unknown";
}

static json tensor_buffer(const ggml_tensor * tensor) {
    if (tensor->buffer == nullptr) {
        return nullptr;
    }

    const ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(tensor->buffer);
    const ggml_backend_dev_t         device      = ggml_backend_buft_get_device(buffer_type);

    json result = {
        { "name",  ggml_backend_buffer_name(tensor->buffer)                         },
        { "type",  ggml_backend_buft_name(buffer_type)                              },
        { "usage", buffer_usage_name(ggml_backend_buffer_get_usage(tensor->buffer)) },
    };
    if (device != nullptr) {
        result["device"] = ggml_backend_dev_name(device);
    }
    return result;
}

static json tensor_backend(const llama_context * ctx, const ggml_tensor * tensor) {
    const char * name = llama_graph_tensor_backend_name(ctx, tensor);
    return name == nullptr ? json(nullptr) : json(name);
}

static std::string tensor_kind(const ggml_tensor * tensor, bool is_graph_node) {
    if (is_graph_node) {
        return "operation";
    }
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        return "input";
    }
    if (tensor->buffer != nullptr &&
        ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return "weight";
    }
    return "leaf";
}

struct physical_plan_export_context {
    const std::unordered_map<const ggml_tensor *, size_t> * tensor_ids;
    const std::unordered_map<const ggml_tensor *, int> *    graph_order;
    json                                                     slots = json::array();
    std::string                                              error;
};

static bool physical_plan_tensor_id(physical_plan_export_context * context,
                                    const ggml_tensor *             tensor,
                                    std::string *                   result) {
    const auto found = context->tensor_ids->find(tensor);
    if (found == context->tensor_ids->end()) {
        context->error = "DAYAN physical slot refers to a tensor outside the exported graph";
        return false;
    }
    *result = "n" + std::to_string(found->second);
    return true;
}

static void physical_plan_callback(const ggml_backend_dayan_graph_plan_slot * slot, void * user_data) {
    physical_plan_export_context * context = (physical_plan_export_context *) user_data;
    if (!context->error.empty()) {
        return;
    }
    if (slot->version != 1) {
        context->error = "unsupported DAYAN physical graph plan version";
        return;
    }

    json logical_nodes = json::array();
    int  order_start   = std::numeric_limits<int>::max();
    int  order_end     = -1;
    for (size_t i = 0; i < GGML_DAYAN_SLOT_TRACE_MAX_NODES; ++i) {
        const ggml_tensor * node = slot->nodes[i];
        if (node == nullptr) {
            continue;
        }
        std::string id;
        if (!physical_plan_tensor_id(context, node, &id)) {
            return;
        }
        json item = {
            { "slot_node_index", i                  },
            { "tensor",          id                 },
            { "name",            node->name         },
            { "op",              ggml_op_name(node->op) },
        };
        const auto order = context->graph_order->find(node);
        if (order != context->graph_order->end()) {
            item["order"] = order->second;
            order_start   = std::min(order_start, order->second);
            order_end     = std::max(order_end, order->second);
        } else {
            item["order"] = nullptr;
        }
        logical_nodes.push_back(std::move(item));
    }

    json inputs = json::array();
    for (size_t i = 0; i < slot->input_count; ++i) {
        std::string id;
        if (!physical_plan_tensor_id(context, slot->inputs[i], &id)) {
            return;
        }
        inputs.push_back({
            { "input_index", i                    },
            { "role",        slot->input_roles[i] },
            { "tensor",      id                   },
        });
    }

    json outputs = json::array();
    for (size_t i = 0; i < slot->output_count; ++i) {
        std::string id;
        if (!physical_plan_tensor_id(context, slot->outputs[i], &id)) {
            return;
        }
        outputs.push_back({
            { "output_index", i  },
            { "tensor",       id },
        });
    }

    context->slots.push_back({
        { "id",           "s" + std::to_string(slot->slot_index) },
        { "index",        slot->slot_index                         },
        { "kind",         slot->kind                               },
        { "node_count",   slot->node_count                         },
        { "logical_nodes", std::move(logical_nodes)                },
        { "inputs",       std::move(inputs)                        },
        { "outputs",      std::move(outputs)                       },
        { "output_mask",  slot->output_mask                        },
        { "order_start",  order_end >= 0 ? json(order_start) : json(nullptr) },
        { "order_end",    order_end >= 0 ? json(order_end) : json(nullptr)   },
        { "implementation",
         {
             { "executor",    slot->executor    },
             { "launch_name", slot->launch_name },
             { "kernel",      slot->kernel      },
             { "task_mode",   slot->task_mode   },
             { "task_dim_x",  slot->task_dim_x  },
         } },
    });
}

static json export_physical_plan(ggml_backend_dayan_get_graph_plan_t                  get_plan,
                                 const ggml_cgraph *                                  graph,
                                 const std::unordered_map<const ggml_tensor *, size_t> & tensor_ids,
                                 const std::unordered_map<const ggml_tensor *, int> &    graph_order) {
    if (get_plan == nullptr) {
        return nullptr;
    }

    physical_plan_export_context context;
    context.tensor_ids  = &tensor_ids;
    context.graph_order = &graph_order;
    if (!get_plan(graph, physical_plan_callback, &context)) {
        throw std::runtime_error("DAYAN failed to compile the physical graph plan");
    }
    if (!context.error.empty()) {
        throw std::runtime_error(context.error);
    }
    return {
        { "backend",    GGML_DAYAN_NAME      },
        { "slot_count", context.slots.size() },
        { "slots",      std::move(context.slots) },
    };
}

static json export_graph(llama_context * ctx,
                         ggml_backend_dayan_get_graph_plan_t get_dayan_plan,
                         const char *    name,
                         uint32_t        n_tokens,
                         uint32_t        n_seqs,
                         uint32_t        n_outputs) {
    ggml_cgraph * graph = llama_graph_reserve_split(ctx, n_tokens, n_seqs, n_outputs);
    if (graph == nullptr) {
        throw std::runtime_error(std::string("failed to reserve graph: ") + name);
    }

    const int                        n_graph_nodes = ggml_graph_n_nodes(graph);
    std::vector<const ggml_tensor *> tensors;
    tensors.reserve(n_graph_nodes * 2);

    std::unordered_map<const ggml_tensor *, size_t> tensor_ids;
    std::unordered_map<const ggml_tensor *, int>    graph_order;
    std::unordered_set<const ggml_tensor *>         graph_nodes;

    const auto add_tensor = [&](const ggml_tensor * tensor) {
        if (tensor != nullptr && tensor_ids.emplace(tensor, tensor_ids.size()).second) {
            tensors.push_back(tensor);
        }
    };

    for (int i = 0; i < n_graph_nodes; ++i) {
        const ggml_tensor * tensor = ggml_graph_node(graph, i);
        add_tensor(tensor);
        graph_order.emplace(tensor, i);
        graph_nodes.insert(tensor);
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor * tensor = tensors[i];
        for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            add_tensor(tensor->src[source_index]);
        }
        add_tensor(tensor->view_src);
    }

    json nodes = json::array();
    json edges = json::array();

    for (const ggml_tensor * tensor : tensors) {
        const size_t tensor_id = tensor_ids.at(tensor);
        const auto   order     = graph_order.find(tensor);

        json node = {
            { "id", "n" + std::to_string(tensor_id) },
            { "name", tensor->name },
            { "kind", tensor_kind(tensor, graph_nodes.count(tensor) != 0) },
            { "op", ggml_op_name(tensor->op) },
            { "op_symbol", ggml_op_symbol(tensor->op) },
            { "type", ggml_type_name(tensor->type) },
            { "shape", tensor_dimensions(tensor->ne) },
            { "strides", tensor_strides(tensor->nb) },
            { "elements", ggml_nelements(tensor) },
            { "bytes", ggml_nbytes(tensor) },
            { "quantized", ggml_is_quantized(tensor->type) },
            { "flags", tensor_flags(tensor) },
            { "buffer", tensor_buffer(tensor) },
            { "execution_backend", tensor_backend(ctx, tensor) },
            { "view_offset", tensor->view_offs },
        };
        if (order != graph_order.end()) {
            node["order"] = order->second;
        } else {
            node["order"] = nullptr;
        }
        if (tensor->view_src != nullptr) {
            node["view_source"] = "n" + std::to_string(tensor_ids.at(tensor->view_src));
        }
        nodes.push_back(std::move(node));

        bool view_is_source = false;
        for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            const ggml_tensor * source = tensor->src[source_index];
            if (source == nullptr) {
                continue;
            }
            view_is_source = view_is_source || source == tensor->view_src;
            edges.push_back({
                { "source", "n" + std::to_string(tensor_ids.at(source)) },
                { "target", "n" + std::to_string(tensor_id)             },
                { "label",  "src " + std::to_string(source_index)       },
                { "kind",   "source"                                    },
            });
        }
        if (tensor->view_src != nullptr && !view_is_source) {
            edges.push_back({
                { "source", "n" + std::to_string(tensor_ids.at(tensor->view_src)) },
                { "target", "n" + std::to_string(tensor_id)                       },
                { "label",  "view"                                                },
                { "kind",   "view"                                                },
            });
        }
    }

    return {
        { "name",            name             },
        { "n_tokens",        n_tokens         },
        { "n_seqs",          n_seqs           },
        { "n_outputs",       n_outputs        },
        { "operation_count", n_graph_nodes    },
        { "tensor_count",    tensors.size()   },
        { "nodes",           std::move(nodes) },
        { "edges",           std::move(edges) },
        { "physical_plan",   export_physical_plan(get_dayan_plan, graph, tensor_ids, graph_order) },
    };
}

static std::string model_description(const llama_model * model) {
    std::vector<char> buffer(256);
    int32_t           length = llama_model_desc(model, buffer.data(), buffer.size());
    if (length < 0) {
        return {};
    }
    if ((size_t) length >= buffer.size()) {
        buffer.resize(length + 1);
        llama_model_desc(model, buffer.data(), buffer.size());
    }
    return buffer.data();
}

static json model_devices(const llama_model * model) {
    json result = json::array();
    for (int32_t i = 0; i < llama_model_n_devices(model); ++i) {
        ggml_backend_dev_t device = llama_model_get_device(model, i);
        if (device != nullptr) {
            result.push_back(ggml_backend_dev_name(device));
        }
    }
    return result;
}

static ggml_backend_dayan_get_graph_plan_t model_dayan_graph_plan_getter(const llama_model * model) {
    for (int32_t i = 0; i < llama_model_n_devices(model); ++i) {
        ggml_backend_dev_t device = llama_model_get_device(model, i);
        if (device == nullptr || std::strcmp(ggml_backend_dev_name(device), GGML_DAYAN_NAME) != 0) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        return (ggml_backend_dayan_get_graph_plan_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_dayan_get_graph_plan");
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.out_file   = "graph.json";
    params.warmup     = false;
    params.fit_params = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS)) {
        return 1;
    }

    params.load_mode      = LLAMA_LOAD_MODE_NONE;
    params.no_extra_bufts = true;
    params.no_alloc       = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto            llama_init = common_init_from_params(params);
    llama_model *   model      = llama_init->model();
    llama_context * ctx        = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to initialize model and context\n");
        return 1;
    }

    const ggml_backend_dayan_get_graph_plan_t get_dayan_plan = model_dayan_graph_plan_getter(model);

    uint32_t n_prefill = std::min(llama_n_ctx_seq(ctx), llama_n_ubatch(ctx));
    if (!params.prompt.empty()) {
        const llama_vocab *            vocab   = llama_model_get_vocab(model);
        const bool                     add_bos = llama_vocab_get_add_bos(vocab);
        const std::vector<llama_token> tokens  = common_tokenize(ctx, params.prompt, add_bos, true);
        if (!tokens.empty()) {
            n_prefill = std::min<uint32_t>(tokens.size(), n_prefill);
        }
    }
    n_prefill = std::max<uint32_t>(n_prefill, 1);

    json output = {
        { "version", 2             },
        { "model",
         {
              { "description", model_description(model) },
              { "size", llama_model_size(model) },
              { "parameters", llama_model_n_params(model) },
              { "layers", llama_model_n_layer(model) },
              { "devices", model_devices(model) },
          }                        },
        { "context",
         {
              { "n_ctx", llama_n_ctx(ctx) },
              { "n_ctx_seq", llama_n_ctx_seq(ctx) },
              { "n_ubatch", llama_n_ubatch(ctx) },
          }                        },
        { "graphs",  json::array() },
    };

    output["graphs"].push_back(export_graph(ctx, get_dayan_plan, "prefill", n_prefill, 1, 1));
    output["graphs"].push_back(export_graph(ctx, get_dayan_plan, "decode", 1, 1, 1));

    std::ofstream file(params.out_file);
    if (!file) {
        LOG_ERR("failed to create output file: %s\n", params.out_file.c_str());
        return 1;
    }
    file << std::setw(2) << output << '\n';
    if (!file) {
        LOG_ERR("failed to write output file: %s\n", params.out_file.c_str());
        return 1;
    }

    LOG_INF("wrote %s\n", params.out_file.c_str());
    llama_backend_free();
    return 0;
}
