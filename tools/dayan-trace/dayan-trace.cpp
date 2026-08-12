#include "common.h"
#include "ggml-backend.h"
#include "ggml-dayan.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

struct trace_options {
    std::string model_path;
    std::string trace_path;
    std::string replay_path;
    int32_t     prefill     = 128;
    bool        prefill_set = false;
    bool        decode      = false;
    bool        check       = false;
};

struct output_digest {
    size_t      bytes = 0;
    std::string hash;
};

struct slot_sample {
    json                       signature;
    std::string                signature_key;
    int64_t                    elapsed_us = 0;
    int32_t                    status     = 0;
    bool                       executed   = false;
    std::vector<output_digest> outputs;
};

struct trace_round {
    std::string              name;
    std::vector<slot_sample> samples;
    int64_t                  setup_us      = 0;
    int64_t                  wall_us       = 0;
    int32_t                  setup_status  = 0;
    int32_t                  decode_status = 0;
    std::string              callback_error;
};

static void print_usage(const char * argv0) {
    std::cout << "usage: " << argv0 << " -m MODEL --trace FILE [--prefill 128] [--decode] [--check]\n"
              << "       " << argv0 << " [-m MODEL] --replay FILE [--trace FILE] [--check]\n";
}

static int32_t parse_positive_i32(const char * value, const char * option) {
    char *     end    = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    return (int32_t) parsed;
}

static trace_options parse_options(int argc, char ** argv) {
    trace_options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--check") {
            options.check = true;
            continue;
        }
        if (argument == "--decode") {
            options.decode = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const char * value = argv[++i];
        if (argument == "-m" || argument == "--model") {
            options.model_path = value;
        } else if (argument == "--trace") {
            options.trace_path = value;
        } else if (argument == "--replay") {
            options.replay_path = value;
        } else if (argument == "--prefill") {
            options.prefill     = parse_positive_i32(value, "--prefill");
            options.prefill_set = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }

    if (options.replay_path.empty() && options.trace_path.empty()) {
        throw std::runtime_error("one of --trace or --replay is required");
    }
    if (options.replay_path.empty() && options.model_path.empty()) {
        throw std::runtime_error("-m is required when capturing a trace");
    }
    if (!options.replay_path.empty() && options.trace_path == options.replay_path) {
        throw std::runtime_error("--trace must not overwrite the replay input");
    }
    return options;
}

static json read_json(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open JSON file: " + path);
    }
    json value;
    input >> value;
    return value;
}

static void write_json(const std::string & path, const json & value) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create JSON file: " + path);
    }
    output << std::setw(2) << value << '\n';
    if (!output) {
        throw std::runtime_error("failed to write JSON file: " + path);
    }
}

static json dimensions(const int64_t * values) {
    json result = json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.push_back(values[i]);
    }
    return result;
}

static json strides(const size_t * values) {
    json result = json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.push_back(values[i]);
    }
    return result;
}

static json tensor_layout(const ggml_tensor * tensor, bool include_view) {
    json result = {
        { "type", ggml_type_name(tensor->type) },
        { "ne",   dimensions(tensor->ne)       },
        { "nb",   strides(tensor->nb)          },
    };
    if (include_view) {
        result["is_view"]   = tensor->view_src != nullptr;
        result["view_offs"] = tensor->view_offs;
        if (tensor->view_src != nullptr) {
            result["view_src"] = tensor_layout(tensor->view_src, false);
        }
    }
    return result;
}

static std::string bytes_to_hex(const void * data, size_t size) {
    static const char digits[] = "0123456789abcdef";
    const uint8_t *   bytes    = (const uint8_t *) data;
    std::string       result(size * 2, '0');
    for (size_t i = 0; i < size; ++i) {
        result[2 * i]     = digits[bytes[i] >> 4];
        result[2 * i + 1] = digits[bytes[i] & 0x0f];
    }
    return result;
}

static json make_signature(const ggml_backend_dayan_slot_trace_event * event) {
    json signature;
    signature["kind"]       = event->kind;
    signature["node_count"] = event->node_count;
    if (event->output_mask != 0U) {
        signature["output_mask"] = event->output_mask;
    }
    signature["nodes"]      = json::array();
    for (size_t i = 0; i < GGML_DAYAN_SLOT_TRACE_MAX_NODES; ++i) {
        const ggml_tensor * node = event->nodes[i];
        if (node == nullptr) {
            continue;
        }

        json sources = json::array();
        for (size_t source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            if (node->src[source_index] != nullptr) {
                sources.push_back({
                    { "index", source_index },
                    { "tensor", tensor_layout(node->src[source_index], true) },
                });
            }
        }
        signature["nodes"].push_back({
            { "slot_node_index", i },
            { "op", ggml_op_name(node->op) },
            { "output", tensor_layout(node, true) },
            { "sources", std::move(sources) },
            { "op_params", bytes_to_hex(node->op_params, GGML_MAX_OP_PARAMS) },
        });
    }

    signature["outputs"] = json::array();
    for (size_t i = 0; i < event->output_count; ++i) {
        if (event->outputs[i] != nullptr) {
            signature["outputs"].push_back(tensor_layout(event->outputs[i], true));
        }
    }
    return signature;
}

static std::string hash_bytes(const uint8_t * data, size_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

class trace_collector {
  public:
    explicit trace_collector(bool check_outputs) : check_outputs_(check_outputs) {}

    void begin(trace_round * round) { active_round_ = round; }

    void end() { active_round_ = nullptr; }

    static void callback(const ggml_backend_dayan_slot_trace_event * event, void * user_data) {
        trace_collector * collector = (trace_collector *) user_data;
        if (collector->active_round_ == nullptr || !collector->active_round_->callback_error.empty()) {
            return;
        }
        try {
            collector->record(event);
        } catch (const std::exception & error) {
            collector->active_round_->callback_error = error.what();
        }
    }

  private:
    void record(const ggml_backend_dayan_slot_trace_event * event) {
        if (event->version != 3) {
            throw std::runtime_error("unsupported Dayan slot trace event version");
        }

        slot_sample sample;
        sample.signature     = make_signature(event);
        sample.signature_key = sample.signature.dump();
        sample.elapsed_us    = event->elapsed_us;
        sample.status        = event->status;
        sample.executed      = event->executed;

        if (check_outputs_ && event->executed && event->status == 0) {
            for (size_t i = 0; i < event->output_count; ++i) {
                const ggml_tensor * output = event->outputs[i];
                if (output == nullptr || output->buffer == nullptr) {
                    throw std::runtime_error("fused slot output has no backend buffer");
                }
                const size_t size = ggml_nbytes(output);
                scratch_.resize(size);
                ggml_backend_tensor_get(output, scratch_.data(), 0, size);
                sample.outputs.push_back({ size, hash_bytes(scratch_.data(), size) });
            }
        }
        active_round_->samples.push_back(std::move(sample));
    }

    bool                 check_outputs_ = false;
    trace_round *        active_round_  = nullptr;
    std::vector<uint8_t> scratch_;
};

class trace_filter {
  public:
    enum class mode {
        skip_all,
        unique,
    };

    void reset(mode value) {
        mode_ = value;
        seen_.clear();
        error_.clear();
    }

    const std::string & error() const { return error_; }

    static bool callback(const ggml_backend_dayan_slot_trace_event * event, void * user_data) {
        trace_filter * filter = (trace_filter *) user_data;
        if (filter->mode_ == mode::skip_all || !filter->error_.empty()) {
            return false;
        }
        try {
            return filter->seen_.insert(make_signature(event).dump()).second;
        } catch (const std::exception & error) {
            filter->error_ = error.what();
            return false;
        }
    }

  private:
    mode                            mode_ = mode::skip_all;
    std::unordered_set<std::string> seen_;
    std::string                     error_;
};

struct callback_guard {
    ggml_backend_dayan_set_slot_trace_callback_t setter = nullptr;
    ggml_backend_dayan_set_slot_filter_callback_t filter_setter = nullptr;

    ~callback_guard() {
        if (setter != nullptr) {
            setter(nullptr, nullptr);
        }
        if (filter_setter != nullptr) {
            filter_setter(nullptr, nullptr);
        }
    }
};

static std::vector<llama_token> make_tokens(const llama_model * model, int32_t prefill) {
    std::string text;
    for (int32_t i = 0; i < prefill; ++i) {
        text += " Dayan fused operator trace";
    }
    std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(model), text, true, false);
    if ((int32_t) tokens.size() < prefill) {
        throw std::runtime_error("failed to create enough deterministic input tokens");
    }
    tokens.resize(prefill);
    return tokens;
}

static trace_round run_round(llama_context *                  ctx,
                             const std::vector<llama_token> & tokens,
                             const trace_options &            options,
                             const std::string &              name,
                             trace_collector *                collector,
                             trace_filter *                   filter) {
    trace_round round;
    round.name = name;
    llama_memory_clear(llama_get_memory(ctx), true);

    if (options.decode) {
        filter->reset(trace_filter::mode::skip_all);
        const int64_t setup_start_us = ggml_time_us();
        round.setup_status = llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(tokens.data()), options.prefill));
        llama_synchronize(ctx);
        round.setup_us = ggml_time_us() - setup_start_us;
        if (round.setup_status != 0) {
            return round;
        }
    }

    llama_token * trace_tokens = const_cast<llama_token *>(tokens.data());
    int32_t       trace_count  = options.prefill;
    if (options.decode) {
        trace_tokens += options.prefill;
        trace_count = 1;
    }
    filter->reset(trace_filter::mode::unique);
    collector->begin(&round);
    const int64_t start_us = ggml_time_us();
    round.decode_status    = llama_decode(ctx, llama_batch_get_one(trace_tokens, trace_count));
    llama_synchronize(ctx);
    round.wall_us = ggml_time_us() - start_us;
    collector->end();
    if (!filter->error().empty()) {
        round.callback_error = "slot filter: " + filter->error();
    }
    return round;
}

static std::vector<std::string> signature_sequence(const trace_round & round) {
    std::vector<std::string> result;
    result.reserve(round.samples.size());
    for (const slot_sample & sample : round.samples) {
        result.push_back(sample.signature_key);
    }
    return result;
}

static std::unordered_map<std::string, size_t> signature_counts(const trace_round & round) {
    std::unordered_map<std::string, size_t> result;
    for (const slot_sample & sample : round.samples) {
        ++result[sample.signature_key];
    }
    return result;
}

static json digest_json(const output_digest & digest) {
    return {
        { "bytes", digest.bytes },
        { "hash",  digest.hash  },
    };
}

static json round_metrics(const std::vector<const slot_sample *> & samples, size_t trace_count, bool include_outputs) {
    json result;
    result["calls"]      = samples.size();
    result["samples_us"] = json::array();
    int64_t total        = 0;
    int64_t minimum      = samples.empty() ? 0 : samples.front()->elapsed_us;
    int64_t maximum      = 0;
    for (const slot_sample * sample : samples) {
        result["samples_us"].push_back(sample->elapsed_us);
        total += sample->elapsed_us;
        minimum = std::min(minimum, sample->elapsed_us);
        maximum = std::max(maximum, sample->elapsed_us);
    }
    const double mean                = samples.empty() ? 0.0 : (double) total / samples.size();
    result["total_us"]               = total;
    result["min_us"]                 = minimum;
    result["max_us"]                 = maximum;
    result["mean_us"]                = mean;
    result["latency_times_count_us"] = mean * trace_count;

    if (include_outputs) {
        result["outputs"] = json::array();
        for (const slot_sample * sample : samples) {
            json call = json::array();
            for (const output_digest & digest : sample->outputs) {
                call.push_back(digest_json(digest));
            }
            result["outputs"].push_back(std::move(call));
        }
    }
    return result;
}

struct slot_aggregate {
    std::string                      id;
    json                             signature;
    std::string                      key;
    size_t                           trace_count = 0;
    std::vector<const slot_sample *> rounds[2];
};

struct reference_data {
    bool                                         enabled = false;
    std::vector<std::string>                     keys;
    std::vector<std::string>                     sequence;
    std::unordered_map<std::string, std::string> ids;
    std::unordered_map<std::string, size_t>      counts;
};

static reference_data load_reference(const json & document) {
    if (document.at("format") != "llama-dayan-fused-slot-trace" || document.at("version") != 1) {
        throw std::runtime_error("unsupported Dayan trace format");
    }

    reference_data reference;
    reference.enabled = true;
    std::unordered_map<std::string, std::string> id_to_key;
    for (const json & slot : document.at("slots")) {
        const std::string key = slot.at("signature").dump();
        const std::string id  = slot.at("id");
        reference.keys.push_back(key);
        reference.ids.emplace(key, id);
        reference.counts.emplace(key, slot.at("count").get<size_t>());
        id_to_key.emplace(id, key);
    }
    if (document.contains("slot_sequence")) {
        for (const json & id_value : document.at("slot_sequence")) {
            const std::string id    = id_value;
            const auto        found = id_to_key.find(id);
            if (found == id_to_key.end()) {
                throw std::runtime_error("trace slot_sequence contains an unknown id: " + id);
            }
            reference.sequence.push_back(found->second);
        }
    }
    return reference;
}

static bool same_counts(const std::unordered_map<std::string, size_t> & left,
                        const std::unordered_map<std::string, size_t> & right) {
    return left == right;
}

static bool same_outputs(const std::vector<const slot_sample *> & cold,
                         const std::vector<const slot_sample *> & warm,
                         size_t *                                 compared,
                         size_t *                                 mismatched) {
    *compared   = 0;
    *mismatched = 0;
    if (cold.size() != warm.size()) {
        *mismatched = std::max(cold.size(), warm.size());
        return false;
    }
    for (size_t i = 0; i < cold.size(); ++i) {
        if (cold[i]->outputs.size() != warm[i]->outputs.size()) {
            ++*mismatched;
            continue;
        }
        for (size_t output = 0; output < cold[i]->outputs.size(); ++output) {
            ++*compared;
            if (cold[i]->outputs[output].bytes != warm[i]->outputs[output].bytes ||
                cold[i]->outputs[output].hash != warm[i]->outputs[output].hash) {
                ++*mismatched;
            }
        }
    }
    return *mismatched == 0;
}

static std::string output_label(const json & signature) {
    if (signature.at("outputs").empty()) {
        return "none";
    }
    const json &       output = signature.at("outputs").front();
    std::ostringstream stream;
    stream << output.at("type").get<std::string>() << '[';
    const json & ne = output.at("ne");
    for (size_t i = 0; i < ne.size(); ++i) {
        if (i != 0) {
            stream << 'x';
        }
        stream << ne[i].get<int64_t>();
    }
    stream << ']';
    return stream.str();
}

static json build_report(const trace_options &            options,
                         const llama_model *              model,
                         const std::vector<llama_token> & tokens,
                         const trace_round (&rounds)[2],
                         const reference_data & reference,
                         bool *                 valid) {
    std::unordered_map<std::string, const json *> signatures;
    for (const trace_round & round : rounds) {
        for (const slot_sample & sample : round.samples) {
            signatures.emplace(sample.signature_key, &sample.signature);
        }
    }

    std::vector<std::string>              keys = reference.keys;
    std::unordered_map<std::string, bool> listed;
    for (const std::string & key : keys) {
        listed.emplace(key, true);
    }
    for (const trace_round & round : rounds) {
        for (const slot_sample & sample : round.samples) {
            if (listed.emplace(sample.signature_key, true).second) {
                keys.push_back(sample.signature_key);
            }
        }
    }

    const std::unordered_map<std::string, size_t>   cold_counts   = signature_counts(rounds[0]);
    const std::unordered_map<std::string, size_t>   warm_counts   = signature_counts(rounds[1]);
    const std::unordered_map<std::string, size_t> & target_counts = reference.enabled ? reference.counts : cold_counts;
    const std::vector<std::string>                  cold_sequence = signature_sequence(rounds[0]);
    const std::vector<std::string>                  warm_sequence = signature_sequence(rounds[1]);
    const std::vector<std::string> &                target_sequence =
        reference.enabled && !reference.sequence.empty() ? reference.sequence : cold_sequence;

    std::vector<slot_aggregate> aggregates;
    aggregates.reserve(keys.size());
    std::unordered_map<std::string, size_t> aggregate_indices;
    for (const std::string & key : keys) {
        slot_aggregate aggregate;
        aggregate.key        = key;
        const auto signature = signatures.find(key);
        aggregate.signature  = signature != signatures.end() ? *signature->second : json::parse(key);
        const auto id        = reference.ids.find(key);
        if (id != reference.ids.end()) {
            aggregate.id = id->second;
        } else {
            std::ostringstream generated;
            generated << "slot_" << std::setfill('0') << std::setw(4) << aggregates.size();
            aggregate.id = generated.str();
        }
        const auto count      = target_counts.find(key);
        aggregate.trace_count = count == target_counts.end() ? 0 : count->second;
        aggregate_indices.emplace(key, aggregates.size());
        aggregates.push_back(std::move(aggregate));
    }
    for (size_t round_index = 0; round_index < 2; ++round_index) {
        for (const slot_sample & sample : rounds[round_index].samples) {
            if (sample.executed) {
                aggregates.at(aggregate_indices.at(sample.signature_key)).rounds[round_index].push_back(&sample);
            }
        }
    }

    char model_description[256] = {};
    llama_model_desc(model, model_description, sizeof(model_description));
    json report;
    report["format"]            = "llama-dayan-fused-slot-trace";
    report["version"]           = 1;
    report["backend"]           = GGML_DAYAN_NAME;
    report["model_path"]        = options.model_path;
    report["model_description"] = model_description;
    report["model_parameters"]  = llama_model_n_params(model);
    report["phase"]             = options.decode ? "decode" : "prefill";
    report["prefill_size"]      = options.prefill;
    report["trace_token_count"] = options.decode ? 1 : options.prefill;
    report["tokens"]            = tokens;
    report["check_outputs"]     = options.check;
    report["check_scope"]       = "exact cold/warm fused-slot output hashes; not an independent CPU reference";
    report["slot_timer_scope"]  = "host preparation, config upload/cache lookup, kernel launch, and device wait";
    report["check_downloads_in_slot_timer"] = false;
    if (reference.enabled) {
        report["replay_source"] = options.replay_path;
    }

    *valid          = true;
    report["slots"] = json::array();
    for (slot_aggregate & aggregate : aggregates) {
        json slot;
        slot["id"]             = aggregate.id;
        slot["signature"]      = aggregate.signature;
        slot["count"]          = aggregate.trace_count;
        slot["rounds"]["cold"] = round_metrics(aggregate.rounds[0], aggregate.trace_count, options.check);
        slot["rounds"]["warm"] = round_metrics(aggregate.rounds[1], aggregate.trace_count, options.check);
        if (options.check) {
            size_t     compared   = 0;
            size_t     mismatched = 0;
            const bool matches    = same_outputs(aggregate.rounds[0], aggregate.rounds[1], &compared, &mismatched);
            slot["check"]         = {
                { "pass",               matches    },
                { "outputs_compared",   compared   },
                { "outputs_mismatched", mismatched },
            };
            *valid = *valid && matches;
        }
        report["slots"].push_back(std::move(slot));
    }

    std::unordered_map<std::string, std::string> key_to_id;
    for (const slot_aggregate & aggregate : aggregates) {
        key_to_id.emplace(aggregate.key, aggregate.id);
    }
    report["slot_sequence"] = json::array();
    for (const std::string & key : cold_sequence) {
        report["slot_sequence"].push_back(key_to_id.at(key));
    }

    const std::unordered_map<std::string, size_t> round_counts[2]    = { cold_counts, warm_counts };
    const std::vector<std::string>                round_sequences[2] = { cold_sequence, warm_sequence };
    for (size_t round_index = 0; round_index < 2; ++round_index) {
        int64_t slot_total_us      = 0;
        double  estimated_total_us = 0.0;
        size_t  executed_slots     = 0;
        bool    all_sampled        = true;
        for (const slot_aggregate & aggregate : aggregates) {
            executed_slots += aggregate.rounds[round_index].size();
            for (const slot_sample * sample : aggregate.rounds[round_index]) {
                slot_total_us += sample->elapsed_us;
            }
            if (!aggregate.rounds[round_index].empty()) {
                int64_t total = 0;
                for (const slot_sample * sample : aggregate.rounds[round_index]) {
                    total += sample->elapsed_us;
                }
                estimated_total_us += (double) total / aggregate.rounds[round_index].size() * aggregate.trace_count;
            } else if (aggregate.trace_count != 0) {
                all_sampled = false;
            }
        }
        const bool counts_match                    = same_counts(round_counts[round_index], target_counts);
        const bool order_matches                   = round_sequences[round_index] == target_sequence;
        const bool callback_ok                     = rounds[round_index].callback_error.empty();
        const bool setup_ok                        = rounds[round_index].setup_status == 0;
        const bool decode_ok                       = rounds[round_index].decode_status == 0;
        report["rounds"][rounds[round_index].name] = {
            { "setup_us",                    rounds[round_index].setup_us                   },
            { "unique_replay_wall_us",       rounds[round_index].wall_us                 },
            { "observed_slots",              rounds[round_index].samples.size()          },
            { "executed_unique_slots",       executed_slots                              },
            { "skipped_duplicate_slots",     rounds[round_index].samples.size() - executed_slots },
            { "measured_slot_total_us",      slot_total_us                               },
            { "estimated_operator_total_us", estimated_total_us                          },
            { "unaccounted_us",              rounds[round_index].wall_us - slot_total_us },
            { "setup_status",                rounds[round_index].setup_status             },
            { "decode_status",               rounds[round_index].decode_status           },
            { "callback_error",              rounds[round_index].callback_error          },
            { "counts_match",                counts_match                                },
            { "order_matches",               order_matches                               },
            { "all_signatures_executed",     all_sampled                                 },
        };
        *valid = *valid && counts_match && order_matches && all_sampled && callback_ok && setup_ok && decode_ok;
    }
    report["validation_pass"] = *valid;
    return report;
}

static void print_report(const json & report) {
    std::cout << "\nFused Dayan slot performance\n";
    for (const json & slot : report.at("slots")) {
        const json & cold = slot.at("rounds").at("cold");
        const json & warm = slot.at("rounds").at("warm");
        std::cout << slot.at("id").get<std::string>() << " kind=" << slot.at("signature").at("kind").get<std::string>()
                  << " output=" << output_label(slot.at("signature")) << " count=" << slot.at("count").get<size_t>()
                  << " cold_mean_us=" << std::fixed << std::setprecision(2) << cold.at("mean_us").get<double>()
                  << " cold_x_count_us=" << cold.at("latency_times_count_us").get<double>()
                  << " warm_mean_us=" << warm.at("mean_us").get<double>()
                  << " warm_x_count_us=" << warm.at("latency_times_count_us").get<double>() << '\n';
    }
    for (const char * name : { "cold", "warm" }) {
        const json & round = report.at("rounds").at(name);
        std::cout << name << " slots=" << round.at("observed_slots").get<size_t>()
                  << " executed_unique=" << round.at("executed_unique_slots").get<size_t>()
                  << " skipped_duplicate=" << round.at("skipped_duplicate_slots").get<size_t>()
                  << " measured_slot_total_ms=" << round.at("measured_slot_total_us").get<int64_t>() / 1000.0
                  << " estimated_operator_total_ms=" << round.at("estimated_operator_total_us").get<double>() / 1000.0
                  << " unique_replay_wall_ms=" << round.at("unique_replay_wall_us").get<int64_t>() / 1000.0
                  << " unaccounted_ms=" << round.at("unaccounted_us").get<int64_t>() / 1000.0
                  << " counts_match=" << round.at("counts_match").get<bool>()
                  << " order_matches=" << round.at("order_matches").get<bool>() << '\n';
    }
    std::cout << "validation_pass=" << report.at("validation_pass").get<bool>() << '\n';
}

int main(int argc, char ** argv) {
    try {
        trace_options            options = parse_options(argc, argv);
        json                     replay_document;
        reference_data           reference;
        std::vector<llama_token> tokens;
        if (!options.replay_path.empty()) {
            replay_document              = read_json(options.replay_path);
            reference                    = load_reference(replay_document);
            const std::string phase      = replay_document.value("phase", "prefill");
            if (phase != "prefill" && phase != "decode") {
                throw std::runtime_error("unsupported trace phase: " + phase);
            }
            if (options.decode && phase != "decode") {
                throw std::runtime_error("--decode does not match the replay trace phase");
            }
            options.decode               = phase == "decode";
            const int32_t traced_prefill = replay_document.at("prefill_size").get<int32_t>();
            if (options.prefill_set && options.prefill != traced_prefill) {
                throw std::runtime_error("--prefill does not match the replay trace");
            }
            options.prefill = traced_prefill;
            tokens          = replay_document.at("tokens").get<std::vector<llama_token>>();
            if (options.model_path.empty()) {
                options.model_path = replay_document.at("model_path").get<std::string>();
            }
        }

        llama_backend_init();
        ggml_backend_load_all();
        ggml_backend_dev_t dayan_device = ggml_backend_dev_by_name(GGML_DAYAN_NAME);
        if (dayan_device == nullptr) {
            throw std::runtime_error("DAYAN backend device is not available");
        }
        ggml_backend_reg_t dayan_reg = ggml_backend_dev_backend_reg(dayan_device);
        auto               setter    = (ggml_backend_dayan_set_slot_trace_callback_t) ggml_backend_reg_get_proc_address(
            dayan_reg, "ggml_backend_dayan_set_slot_trace_callback");
        auto filter_setter = (ggml_backend_dayan_set_slot_filter_callback_t) ggml_backend_reg_get_proc_address(
            dayan_reg, "ggml_backend_dayan_set_slot_filter_callback");
        if (setter == nullptr) {
            throw std::runtime_error("DAYAN backend does not provide fused slot tracing");
        }
        if (filter_setter == nullptr) {
            throw std::runtime_error("DAYAN backend does not provide fused slot filtering");
        }

        ggml_backend_dev_t devices[]    = { dayan_device, nullptr };
        llama_model_params model_params = llama_model_default_params();
        model_params.devices            = devices;
        model_params.n_gpu_layers       = -1;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(options.model_path.c_str(), model_params), llama_model_free);
        if (!model) {
            throw std::runtime_error("failed to load model: " + options.model_path);
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx                = options.prefill + (options.decode ? 1 : 0);
        context_params.n_batch              = options.prefill;
        context_params.n_ubatch             = options.prefill;
        context_params.n_seq_max            = 1;
        context_params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_params.offload_kqv          = true;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        if (!context) {
            throw std::runtime_error("failed to create llama context");
        }

        if (tokens.empty()) {
            tokens = make_tokens(model.get(), options.prefill + (options.decode ? 1 : 0));
        }
        const int32_t expected_tokens = options.prefill + (options.decode ? 1 : 0);
        if ((int32_t) tokens.size() != expected_tokens) {
            throw std::runtime_error("trace token count does not match the trace phase");
        }

        trace_collector collector(options.check);
        trace_filter    filter;
        callback_guard  guard{ setter, filter_setter };
        setter(trace_collector::callback, &collector);
        filter_setter(trace_filter::callback, &filter);
        trace_round rounds[2] = {
            run_round(context.get(), tokens, options, "cold", &collector, &filter),
            run_round(context.get(), tokens, options, "warm", &collector, &filter),
        };
        setter(nullptr, nullptr);
        filter_setter(nullptr, nullptr);
        guard.setter = nullptr;
        guard.filter_setter = nullptr;

        bool       valid  = false;
        const json report = build_report(options, model.get(), tokens, rounds, reference, &valid);
        print_report(report);
        if (!options.trace_path.empty()) {
            write_json(options.trace_path, report);
            std::cout << "trace_file=" << options.trace_path << '\n';
        }

        context.reset();
        model.reset();
        llama_backend_free();
        return valid ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
