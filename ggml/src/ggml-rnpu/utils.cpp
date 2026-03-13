#include "./utils.hpp"
#include "ggml.h"

#include "../../../vendor/nlohmann/json.hpp"
#include <cstdint>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

// Tansfer a shape pointer to a Shape object
static Shape ptrToShape(const int64_t* shape_ptr) {
    if (shape_ptr) {
        return {shape_ptr[0], shape_ptr[1], shape_ptr[2], shape_ptr[3]};
    }
    return {0, 0, 0, 0}; // For not existed shape, set to all 0
}

int64_t OperatorStatsManager::startRecording() {
    return ggml_time_us();
}

void OperatorStatsManager::stopRecording(
    const std::string& op_name,
    const int64_t* input1_shape,
    const int64_t* input2_shape,
    const int64_t* output_shape,
    int64_t start_time,
    const std::string& device) {

    int64_t end_time = ggml_time_us();
    int64_t duration = end_time - start_time;
    double ms = static_cast<double>(duration) / 1000.0;

    // Create key
    OpCallKey key;
    key.op_name = op_name;
    key.device = device;
    key.input1_shape = ptrToShape(input1_shape);
    key.input2_shape = ptrToShape(input2_shape);
    key.output_shape = ptrToShape(output_shape);

    // std::lock_guard<std::mutex> lock(mutex_);

    // Update statistics information
    stats_map_[key].update(ms);
}

std::string OperatorStatsManager::generateJSONReport() const {
    // std::lock_guard<std::mutex> lock(mutex_);

    // Create root object
    json root;
    root["version"] = "0.1";

    // Add META data
    json metadata;
    metadata["description"] = "Statistics for operator calls in llama.cpp";
    root["metadata"] = metadata;

    // Group statistic information by operator name
    std::unordered_map<std::string, std::vector<json>> operator_groups;

    for (const auto& [key, stats] : stats_map_) {
        json call_info;

        // Add input shapes
        std::vector<json> input_shapes;
        if (key.input1_shape != Shape{0, 0, 0, 0}) {
            input_shapes.push_back(key.input1_shape);
        }
        if (key.input2_shape != Shape{0, 0, 0, 0}) {
            input_shapes.push_back(key.input2_shape);
        }

        call_info["input_shapes"] = input_shapes;
        call_info["output_shapes"] = std::vector<json>{key.output_shape};
        call_info["call_count"] = stats.call_count;
        call_info["total_time_ms"] = stats.total_time_ms;
        call_info["avg_time_ms"] = stats.avg_time_ms();

        operator_groups[key.op_name + "@" + key.device].push_back(call_info);
    }

    // Build operator statistics helper array
    json operator_statistics = json::array();
    for (const auto& [name, calls] : operator_groups) {
        json op_stat;
        size_t pos = name.find('@');
        std::string op_name = name.substr(0, pos);
        std::string dev_name = name.substr(pos + 1);
        op_stat["operator_name"] = op_name;
        op_stat["device"] = dev_name;
        op_stat["calls"] = calls;
        operator_statistics.push_back(op_stat);
    }

    root["operator_statistics"] = operator_statistics;

    // Indent 2 spaces
    return root.dump(4);
}

bool OperatorStatsManager::saveReportToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << generateJSONReport();
    file.close();

    return true;
}

void OperatorStatsManager::clear() {
    // std::lock_guard<std::mutex> lock(mutex_);
    stats_map_.clear();
}
