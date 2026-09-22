#include "ggml-backend.h"
#include "llama.h"
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

struct run_result {
    std::vector<std::vector<float>> logits;
    std::vector<uint64_t> launches;
    double wall_s = 0;
};

static run_result run(llama_model * model, launch_counter counter, const json & spec, const char * tag, bool mega) {
    setenv("GGML_DAYAN_MEGA", mega ? "1" : "0", 1);
    unsetenv("GGML_DAYAN_MEGA_TAG");
    const int tokens = spec.at("m"), steps = spec.value("decode_steps", 2) + 1;
    const bool decode = spec.value("phase", "prefill") == "decode";
    auto params = llama_context_default_params();
    params.n_ctx = std::max(256, tokens + steps + 32);
    params.n_batch = params.n_ubatch = std::max(tokens, 1);
    params.n_seq_max = 1;
    params.n_threads = params.n_threads_batch = 8;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free);
    if (!ctx) { throw std::runtime_error("cannot create model context"); }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    run_result result;
    for (int step = 0; step < steps; ++step) {
        const int count = step == 0 ? tokens : 1;
        std::vector<llama_token> input(count);
        for (int i = 0; i < count; ++i) { input[i] = 3 + (i + step + spec.value("seed", 1U)) % (vocab - 3); }
        const bool measured = mega && step == (decode ? steps - 1 : 0);
        if (measured) { setenv("GGML_DAYAN_MEGA_TAG", tag, 1); }
        const uint64_t before = counter();
        const auto start = std::chrono::steady_clock::now();
        if (llama_decode(ctx.get(), llama_batch_get_one(input.data(), count)) != 0) {
            throw std::runtime_error("model inference failed");
        }
        llama_synchronize(ctx.get());
        if (measured) { result.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
        unsetenv("GGML_DAYAN_MEGA_TAG");
        result.launches.push_back(counter() - before);
        if (result.launches.back() != (mega ? (uint64_t) llama_model_n_layer(model) : 0)) {
            throw std::runtime_error("unexpected mega count at step " + std::to_string(step) + ": " + std::to_string(result.launches.back()));
        }
        const float * logits = llama_get_logits(ctx.get());
        result.logits.emplace_back(logits, logits + vocab);
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--capabilities") == 0) {
        std::puts("{\"case_schema\":3,\"cmp\":true,\"dtypes\":[\"mxfp4\"],\"ops\":[\"mega\"],\"comparison_reference\":\"dayan-standalone\"}");
        return 0;
    }
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s CASE.json RESULT.json TAG (DAYAN_MEGA_MODEL or DAYAN_MOE_MODEL selects the fixture)\n", argv[0]);
        return 1;
    }
    json result = {{"status", "failed"}, {"samples", json::array()}};
    int status = 1;
    llama_backend_init();
    try {
        std::ifstream input(argv[1]);
        const json spec = json::parse(input);
        result["case"] = spec;
        if (spec.value("op", "") != "mega" || spec.value("mode", "") != "hybrid" || spec.value("layers", 0) < 1 ||
            spec.value("dtype", "") != "mxfp4" || spec.value("m", 0) < 1 || spec.value("m", 0) > 1024 ||
            spec.value("repeats", 1) != 1 || spec.value("warmup", 0) != 0 ||
            spec.value("decode_steps", 2) < 2 || spec.value("decode_steps", 2) > 16 || !argv[3][0] || std::strlen(argv[3]) > 15) {
            throw std::runtime_error("expected a hybrid case with at least two follow-up decode steps");
        }
        const char * path = std::getenv("DAYAN_MEGA_MODEL");
        if (!path || !path[0]) { path = std::getenv("DAYAN_MOE_MODEL"); }
        if (!path || !path[0]) { throw std::runtime_error("set DAYAN_MEGA_MODEL to an MXFP4 model or fixture"); }
        ggml_backend_load_all();
        ggml_backend_dev_t device = ggml_backend_dev_by_name("DAYAN");
        if (!device) { throw std::runtime_error("Dayan backend is unavailable"); }
        auto counter = (launch_counter) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_dayan_get_mega_launch_count");
        if (!counter) { throw std::runtime_error("Dayan mega counter is unavailable"); }
        ggml_backend_dev_t devices[] = {device, nullptr};
        auto params = llama_model_default_params();
        params.devices = devices;
        params.n_gpu_layers = -1;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path, params), llama_model_free);
        if (!model || llama_model_n_layer(model.get()) != spec.at("layers").get<int>()) {
            throw std::runtime_error("model layer count does not match the case");
        }
        char architecture[64];
        if (llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture)) < 0 ||
            (std::strcmp(architecture, "qwen35moe") != 0 && std::strcmp(architecture, "qwen35") != 0)) { throw std::runtime_error("fixture must use qwen35 or qwen35moe"); }
        const auto reference_start = std::chrono::steady_clock::now();
        const run_result reference = run(model.get(), counter, spec, argv[3], false);
        result["reference_wall_s"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - reference_start).count();
        const run_result actual = run(model.get(), counter, spec, argv[3], true);
        double max_abs = 0, squared_error = 0, squared_reference = 0;
        size_t checked = 0, mismatches = 0;
        json steps = json::array();
        for (size_t step = 0; step < actual.logits.size(); ++step) {
            double step_max = 0;
            for (size_t i = 0; i < actual.logits[step].size(); ++i) {
                const double a = actual.logits[step][i], b = reference.logits[step][i], error = std::abs(a - b);
                if (!std::isfinite(a) || !std::isfinite(b)) { throw std::runtime_error("nonfinite logits"); }
                mismatches += error > 1e-5 + 1e-5 * std::abs(b);
                step_max = std::max(step_max, error);
                squared_error += error * error;
                squared_reference += b * b;
                ++checked;
            }
            max_abs = std::max(max_abs, step_max);
            steps.push_back({{"step", step}, {"tokens", step == 0 ? spec.at("m").get<int>() : 1}, {"mega_launches", actual.launches[step]}, {"max_abs_error", step_max}});
        }
        result["model"] = path;
        result["steps"] = steps;
        result["mega_launched"] = true;
        result["validation"] = "mega versus standalone Dayan, including initial and consecutive decode graphs; CPU operator checks are separate";
        result["checked_elements"] = checked;
        result["max_abs_error"] = max_abs;
        result["comparison"] = {{"status", mismatches ? "fail" : "pass"}, {"checked_elements", checked}, {"mismatched_elements", mismatches},
                                {"max_abs_error", max_abs}, {"relative_l2_error", std::sqrt(squared_error / std::max(squared_reference, 1e-30))}};
        result["samples"] = {{{"tag", argv[3]}, {"warmup", false}, {"wall_s", actual.wall_s}, {"tasks", 8}, {"status", mismatches ? 1 : 0}}};
        result["status"] = mismatches ? "failed" : "ok";
        status = mismatches ? 1 : 0;
    } catch (const std::exception & error) {
        result["error"] = error.what();
        std::fprintf(stderr, "%s\n", error.what());
    }
    llama_backend_free();
    std::ofstream output(argv[2]);
    output << result.dump(2) << '\n';
    return output.good() ? status : 1;
}
