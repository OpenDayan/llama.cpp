#include "layer-compare.h"

#include <limits>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static layer_logit_compare compare(std::initializer_list<float> observed, std::initializer_list<float> reference) {
    layer_logit_compare result;
    auto expected = reference.begin();
    for (float value : observed) { result.add(value, *expected++, 0.03, 0.03); }
    return result;
}

int main() {
    try {
        const auto equal = compare({1, -2, 0}, {1, -2, 0});
        require(equal.passes(false, 0.35, 0.93) && equal.passes(true, 0.35, 0.93), "equal logits failed");

        const auto quantized = compare({0.8f, -1.6f, 0}, {1, -2, 0});
        require(!quantized.passes(false, 0.35, 0.93), "strict check ignored elementwise mismatches");
        require(quantized.passes(true, 0.35, 0.93), "MXFP4 check rejected bounded quantization error");
        require(quantized.report().at("mismatched_elements") == 2, "MXFP4 check lost diagnostic mismatches");

        const auto scaled = compare({0.6f, -1.2f, 0}, {1, -2, 0});
        require(!scaled.passes(true, 0.35, 0.93), "high cosine concealed excessive L2 error");
        require(scaled.passes(true, 0.5, 0.93), "L2 override ignored");
        const auto rotated = compare({0, 1}, {1, 0});
        require(!rotated.passes(true, 2, 0.93), "L2 override disabled cosine check");
        require(rotated.passes(true, 2, 0), "cosine override ignored");

        require(compare({0, 0}, {0, 0}).passes(true, 0, 1), "equal zero logits failed");
        require(!compare({0, 0}, {1, 2}).passes(true, 0.35, 0.93), "zero output passed");
        require(!compare({1, 2}, {0, 0}).passes(true, 0.35, 0.93), "nonzero output passed zero reference");
        require(!compare({}, {}).passes(true, 0.35, 0.93), "empty comparison passed");

        for (float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            for (bool invalid_reference : {false, true}) {
                bool rejected = false;
                try { compare({invalid_reference ? 0 : invalid}, {invalid_reference ? invalid : 0}); }
                catch (const std::runtime_error &) { rejected = true; }
                require(rejected, "nonfinite logits accepted");
            }
        }
        std::puts("PASS: strict and MXFP4 logit comparison");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
