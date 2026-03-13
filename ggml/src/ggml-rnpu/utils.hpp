#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <array>
#include <unordered_map>
#include <mutex>
#include <functional> // for std::hash

// Tensor shape
using Shape = std::array<int64_t, 4>;

// Key of op shape
struct OpCallKey {
    std::string op_name;
    std::string device;
    Shape input1_shape;
    Shape input2_shape; // For single input op, set to all 0
    Shape output_shape; // For no output op, set to all 0

    bool operator==(const OpCallKey& other) const {
        return op_name == other.op_name &&
               input1_shape == other.input1_shape &&
               input2_shape == other.input2_shape &&
               output_shape == other.output_shape;
    }
};

// Hash specification for OpCallKey
namespace std {
    template<>
    struct hash<OpCallKey> {
        size_t operator()(const OpCallKey& key) const {
            size_t h1 = std::hash<std::string>{}(key.op_name);
            size_t h2 = std::hash<std::string>{}(key.device);
            size_t h3 = 0;

            // Hash value for composed shape
            for (int64_t dim : key.input1_shape) {
                h3 = h3 * 31 + std::hash<int64_t>{}(dim);
            }

            size_t h4 = 0;
            for (int64_t dim : key.input2_shape) {
                h4 = h4 * 31 + std::hash<int64_t>{}(dim);
            }

            size_t h5 = 0;
            for (int64_t dim : key.output_shape) {
                h5 = h5 * 31 + std::hash<int64_t>{}(dim);
            }

            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
        }
    };
}

// Structure for operator calling information
struct OperatorStats {
    int call_count = 0;
    double total_time_ms = 0.0;

    // Update statistics information
    void update(double duration_ms) {
        call_count++;
        total_time_ms += duration_ms;
    }

    // Get average operator running time
    double avg_time_ms() const {
        return call_count > 0 ? total_time_ms / call_count : 0.0;
    }
};

// Class for managing all operators statistics
class OperatorStatsManager {
public:
    static OperatorStatsManager& getInstance() {
        static OperatorStatsManager instance;
        return instance;
    }

    // Start recording a new operator call
    int64_t startRecording();

    // End recording of an operator call
    void stopRecording(
        const std::string& op_name,
        const int64_t* input1_shape, // Points to an array of 4 ints
        const int64_t* input2_shape, // May be nullptr
        const int64_t* output_shape, // Points to an array of 4 ints
        int64_t start_time,
        const std::string& device);

    // Generate a JSON format report
    std::string generateJSONReport() const;

    // Save statistics information into a JSON file
    bool saveReportToFile(const std::string& filename) const;

    // Clear statistics information
    void clear();

    // No copy / assignment
    OperatorStatsManager(const OperatorStatsManager&) = delete;
    void operator=(const OperatorStatsManager&) = delete;

private:
    OperatorStatsManager() = default;
    ~OperatorStatsManager() = default;

    mutable std::mutex mutex_;
    std::unordered_map<OpCallKey, OperatorStats> stats_map_;
};

#define RECORD_OP_START() \
    OperatorStatsManager::getInstance().startRecording()

#define RECORD_OP_END(op_name, input1_shape, input2_shape, output_shape, start_time, device) \
    OperatorStatsManager::getInstance().stopRecording(op_name, input1_shape, input2_shape, output_shape, start_time, device)
