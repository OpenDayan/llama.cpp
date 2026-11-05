#include "ggml-backend.h"
#include "llama.h"
#include "layer-compare.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using launch_counter = uint64_t (*)();

static json graph_schemas() {
    json result;
    json slots;
    const char * name = nullptr;
#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) name = #Name; slots = json::array();
#define MEGA_STEP(Name, Kind, Function, Barriers, Mode, ...) \
    slots.push_back({{"name", #Name}, {"kind", #Kind}, {"function", #Function}, {"barriers", Barriers}, {"mode", #Mode}});
#define MEGA_GRAPH_END() result[name] = slots;
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END
    return result;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--schemas") == 0) {
        std::puts(graph_schemas().dump(2).c_str());
        return 0;
    }
    const bool cpu_reference = argc == 5 && std::strcmp(argv[3], "--cpu-reference") == 0;
    const bool prepare_state = argc == 4 && std::strcmp(argv[3], "--prepare-state") == 0;
    const bool compare_ops = argc == 4 && std::strcmp(argv[3], "--cmp-ops") == 0;
    const bool cpu = cpu_reference || prepare_state;
    if (argc != 3 && !cpu && !compare_ops) {
        std::fprintf(stderr, "usage: %s CASE.json RESULT.json [--cpu-reference RUN.json | --prepare-state | --cmp-ops] | --schemas\n", argv[0]);
        return 1;
    }
    json result = {{"schema", 1}, {"status", "failed"}, {"samples", json::array()}};
    int status = 1;
    std::unique_ptr<layer_operator_compare> operators;
    llama_backend_init();
    try {
        std::ifstream input(argv[1]);
        const json spec = json::parse(input);
        result["case"] = spec;
        const int input_len = spec.at("input_len"), output_len = spec.at("output_len");
        const std::string mode = spec.at("mode");
        const bool compare = spec.value("cmp", false) && !compare_ops && !prepare_state;
        const double atol = spec.value("cmp_atol", 0.03), rtol = spec.value("cmp_rtol", 0.03);
        const std::string profile = spec.value("cmp_profile", "strict");
        const double max_rel_l2 = spec.value("cmp_max_rel_l2", 0.35), min_cosine = spec.value("cmp_min_cosine", 0.93);
        if (input_len < 1 || output_len < 0 || int64_t(input_len) + output_len > INT32_MAX - 256 ||
            (mode != "prefill" && mode != "decode") || (mode == "prefill" ? output_len != 0 : output_len == 0)) {
            throw std::runtime_error("invalid mode or input/output lengths");
        }
        if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0 || (cpu_reference && !compare) ||
            (profile != "strict" && profile != "mxfp4") || !std::isfinite(max_rel_l2) || max_rel_l2 < 0 ||
            !std::isfinite(min_cosine) || min_cosine < -1 || min_cosine > 1) {
            throw std::runtime_error("invalid CPU comparison options");
        }
        json actual;
        std::ifstream actual_logits;
        std::ofstream saved_logits;
        if (cpu_reference) {
            std::ifstream file(argv[4]);
            actual = json::parse(file);
            if (actual.at("case") != spec || actual.value("inference_status", actual.value("status", "")) != "ok" || actual.value("backend", "") != "DAYAN") {
                throw std::runtime_error("CPU reference requires a successful Dayan run of the same case");
            }
            actual_logits.open(actual.at("logits_file").get<std::string>(), std::ios::binary);
            if (!actual_logits) { throw std::runtime_error("cannot open saved Dayan logits"); }
        } else if (compare || prepare_state) {
            const std::string path = std::string(argv[2]) + ".logits.f32";
            saved_logits.open(path, std::ios::binary);
            if (!saved_logits) { throw std::runtime_error("cannot create Dayan logits file"); }
            result["logits_file"] = path;
            result["logits_format"] = "native-f32";
        }
        result["backend"] = cpu ? "CPU" : "DAYAN";
        result["slots"] = graph_schemas().at(spec.at("graph").get<std::string>());
        ggml_backend_load_all();
        if (compare_ops) {
            operators.reset(new layer_operator_compare);
            operators->atol = spec.value("op_atol", 1e-4);
            operators->rtol = spec.value("op_rtol", 1e-3);
            if (!std::isfinite(operators->atol) || !std::isfinite(operators->rtol) || operators->atol < 0 || operators->rtol < 0) {
                throw std::runtime_error("invalid operator tolerances");
            }
        }
        if (cpu) {
            // ACCEL buffers can be selected even with zero GPU layers.
            for (size_t i = ggml_backend_reg_count(); i > 0; --i) {
                ggml_backend_reg_t reg = ggml_backend_reg_get(i - 1);
                if (std::strcmp(ggml_backend_reg_name(reg), "CPU") != 0) { ggml_backend_unload(reg); }
            }
        }
        ggml_backend_dev_t device = nullptr;
        launch_counter counter = nullptr;
        if (!cpu) {
            device = ggml_backend_dev_by_name("DAYAN");
            if (!device) { throw std::runtime_error("Dayan backend is unavailable"); }
            if (operators) { operators->attach(device); }
            if (!compare_ops) {
                counter = (launch_counter) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_dayan_get_mega_launch_count");
                if (!counter) { throw std::runtime_error("Dayan mega counter is unavailable"); }
            }
        }
        setenv("GGML_DAYAN_MEGA", cpu || compare_ops ? "0" : "1", 1);
        if (!cpu) {
            std::fprintf(stderr, "%s\n", compare_ops ?
                         "DAYAN execution: standalone slots for CPU operator comparison; mega disabled" :
                         "DAYAN execution: one layer mega per step (lp-prefill/lp-d...); decode loads CPU prompt state");
        }
        ggml_backend_dev_t devices[] = {device, nullptr};
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = cpu ? 0 : -1;
        const std::string path = spec.at("model");
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(), model_params), llama_model_free);
        if (!model || llama_model_n_layer(model.get()) != 1) {
            throw std::runtime_error("expected a one-layer GGUF from layer-perf.py prepare");
        }
        auto params = llama_context_default_params();
        params.n_ctx = std::max(256, input_len + output_len + 32);
        params.n_batch = params.n_ubatch = input_len;
        params.n_seq_max = 1;
        params.n_threads = params.n_threads_batch = 8;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.offload_kqv = !cpu;
        params.op_offload = !cpu;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), params), llama_free);
        if (!ctx) { throw std::runtime_error("cannot create model context"); }
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (vocab <= 3) { throw std::runtime_error("model has no usable vocabulary"); }
        result["vocab_size"] = vocab;
        if (!cpu) {
            auto cores = (launch_counter) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_dayan_get_core_count");
            if (!cores || (cores() != 1 && cores() != 4)) { throw std::runtime_error("layer profiling requires one or four cores"); }
            result["core_count"] = cores();
        }
        if (cpu_reference && actual.at("vocab_size").get<int>() != vocab) {
            throw std::runtime_error("CPU and Dayan vocabulary sizes differ");
        }
        std::vector<float> observed(cpu_reference ? vocab : 0);
        json comparisons = json::array();
        layer_logit_compare total;
        size_t failures = 0;
        std::vector<llama_token> tokens(input_len);
        for (int i = 0; i < input_len; ++i) { tokens[i] = 3 + (uint64_t(i) + spec.value("seed", 1U)) % (vocab - 3); }
        const bool cached_prefill = cpu_reference && mode == "prefill";
        std::vector<float> prompt_logits;
        if (!prepare_state && (mode == "decode" || cached_prefill)) {
            const std::string prompt_path = spec.at("prompt_state");
            std::ifstream file(prompt_path);
            const json prompt = json::parse(file);
            if (prompt.value("status", "") != "ok" || prompt.value("backend", "") != "CPU" || prompt.at("vocab_size").get<int>() != vocab) {
                throw std::runtime_error("invalid CPU prompt checkpoint");
            }
            for (const char * key : {"model", "graph", "source_layer", "input_len", "seed"}) {
                if (prompt.at("case").at(key) != spec.at(key)) { throw std::runtime_error("CPU prompt checkpoint does not match the case"); }
            }
            prompt_logits.resize(vocab);
            std::ifstream logits_file(prompt.at("logits_file").get<std::string>(), std::ios::binary);
            logits_file.read(reinterpret_cast<char *>(prompt_logits.data()), vocab * sizeof(float));
            if (!logits_file || logits_file.peek() != std::char_traits<char>::eof()) { throw std::runtime_error("invalid CPU prompt logits file"); }
            for (float value : prompt_logits) {
                if (!std::isfinite(value)) { throw std::runtime_error("nonfinite CPU prompt logits"); }
            }
            const llama_token next = prompt.at("samples").at(0).at("next_token");
            if (next != std::max_element(prompt_logits.begin(), prompt_logits.end()) - prompt_logits.begin()) {
                throw std::runtime_error("CPU prompt continuation token does not match its logits");
            }
            std::vector<llama_token> restored(input_len);
            size_t count = 0;
            const auto start = std::chrono::steady_clock::now();
            if (!llama_state_load_file(ctx.get(), prompt.at("state_file").get<std::string>().c_str(), restored.data(), restored.size(), &count) ||
                count != tokens.size() || restored != tokens || llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) != input_len - 1) {
                throw std::runtime_error("cannot restore CPU prompt state; prefill will not be recomputed");
            }
            llama_synchronize(ctx.get());
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            result["initial_state"] = {{"backend", "CPU"}, {"checkpoint", prompt_path}, {"tokens", input_len}, {"next_token", next}, {"host_load_wall_s", wall}};
            std::fprintf(stderr, "%s restored CPU prompt: tokens=%d next=%d wall=%.6f s\n", cpu ? "CPU" : "DAYAN", input_len, next, wall);
            if (mode == "decode") { tokens.assign(1, next); }
        }
        const int first_step = mode == "decode" && !prepare_state ? 1 : 0;
        const int last_step = prepare_state ? 0 : output_len;
        for (int step = first_step; step <= last_step; ++step) {
            if (operators) { operators->step = step; }
            char tag[16];
            if (step == 0) { std::snprintf(tag, sizeof(tag), "lp-prefill"); }
            else { std::snprintf(tag, sizeof(tag), "lp-d%010d", step); }
            if (!cpu) { setenv("GGML_DAYAN_MEGA_TAG", tag, 1); }
            const uint64_t before = counter ? counter() : 0;
            const auto start = std::chrono::steady_clock::now();
            if (!cached_prefill && llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
                throw std::runtime_error("model inference failed at step " + std::to_string(step));
            }
            llama_synchronize(ctx.get());
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            unsetenv("GGML_DAYAN_MEGA_TAG");
            if (counter && counter() - before != 1) {
                throw std::runtime_error("expected exactly one mega launch per inference step; fallback or microbatch splitting occurred");
            }
            const float * logits = cached_prefill ? prompt_logits.data() : llama_get_logits_ith(ctx.get(), -1);
            if (!logits) { throw std::runtime_error("missing output logits"); }
            for (int i = 0; i < vocab; ++i) {
                if (!std::isfinite(logits[i])) { throw std::runtime_error("nonfinite output logits"); }
            }
            const llama_token greedy = std::max_element(logits, logits + vocab) - logits;
            llama_token next = greedy;
            if (cpu_reference) {
                actual_logits.read(reinterpret_cast<char *>(observed.data()), vocab * sizeof(float));
                if (!actual_logits) { throw std::runtime_error("truncated Dayan logits file"); }
                layer_logit_compare measured_logits;
                for (int i = 0; i < vocab; ++i) {
                    measured_logits.add(observed[i], logits[i], atol, rtol);
                    total.add(observed[i], logits[i], atol, rtol);
                }
                const json & sample = actual.at("samples").at(mode == "decode" ? step - 1 : 0);
                if (sample.at("step").get<int>() != step) { throw std::runtime_error("Dayan token sequence has a missing step"); }
                next = sample.at("next_token").get<llama_token>();
                if (next < 0 || next >= vocab) { throw std::runtime_error("invalid Dayan continuation token"); }
                const bool passed = measured_logits.passes(profile == "mxfp4", max_rel_l2, min_cosine);
                json comparison = measured_logits.report();
                comparison.update({{"step", step}, {"phase", step == 0 ? "prefill" : "decode"}, {"measured", true},
                                   {"status", passed ? "pass" : "fail"}, {"cpu_greedy_token", greedy}, {"dayan_greedy_token", next}});
                comparisons.push_back(comparison);
                failures += !passed;
            } else if (compare || prepare_state) {
                saved_logits.write(reinterpret_cast<const char *>(logits), vocab * sizeof(float));
                if (!saved_logits) { throw std::runtime_error("cannot save Dayan logits"); }
            }
            json sample = {{"tag", tag}, {"step", step}, {"tokens", tokens.size()},
                           {"context_before", step == 0 ? 0 : input_len + step - 1},
                           {"host_inference_wall_s", wall}, {"mega_launches", counter ? 1 : 0}, {"next_token", next}};
            result["samples"].push_back(sample);
            if (prepare_state) {
                const std::string state_path = std::string(argv[2]) + ".state";
                if (!llama_state_save_file(ctx.get(), state_path.c_str(), tokens.data(), tokens.size())) {
                    throw std::runtime_error("cannot save CPU prompt state");
                }
                result["state_file"] = state_path;
            }
            std::fprintf(stderr, "%s %s: tokens=%zu context=%d mega=%d wall=%.6f s%s\n", cpu ? "CPU" : "DAYAN", tag, tokens.size(), step == 0 ? 0 : input_len + step - 1, counter ? 1 : 0, wall, cached_prefill ? " (cached)" : "");
            tokens.assign(1, next);
        }
        if (cpu_reference) {
            if (actual_logits.peek() != std::char_traits<char>::eof()) { throw std::runtime_error("extra data in Dayan logits file"); }
            result["comparison"] = total.report();
            result["comparison"].update({{"reference", "CPU"}, {"status", failures ? "fail" : "pass"}, {"profile", profile},
                                         {"atol", atol}, {"rtol", rtol}, {"max_relative_l2_error", max_rel_l2}, {"min_cosine_similarity", min_cosine},
                                         {"failed_steps", failures}, {"steps", comparisons}});
        } else if (compare || prepare_state) {
            saved_logits.close();
            if (!saved_logits) { throw std::runtime_error("cannot finish Dayan logits file"); }
        }
        if (operators) { failures += operators->failures + operators->nodes.empty(); }
        result["inference_status"] = "ok";
        result["status"] = failures ? "failed" : "ok";
        status = failures ? 1 : 0;
    } catch (const std::exception & error) {
        result["error"] = error.what();
        std::fprintf(stderr, "%s\n", error.what());
    }
    unsetenv("GGML_DAYAN_MEGA_TAG");
    if (operators) {
        result["operator_comparison"] = operators->report();
        if (result.value("inference_status", "") != "ok") { result["operator_comparison"]["status"] = "incomplete"; }
        operators.reset();
    }
    llama_backend_free();
    std::ofstream output(argv[2]);
    output << result.dump(2) << '\n';
    return output.good() ? status : 1;
}
