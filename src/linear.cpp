#include "pokitlms/ops/linear.hpp"

#include <stdexcept>

namespace pokitlms {

void gemv_f32(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::span<float> output,
    std::size_t in_features,
    std::size_t out_features) {
    if (input.size() != in_features) {
        throw std::invalid_argument("GEMV input size mismatch");
    }
    if (weight.size() != in_features * out_features) {
        throw std::invalid_argument("GEMV weight size mismatch");
    }
    if (!bias.empty() && bias.size() != out_features) {
        throw std::invalid_argument("GEMV bias size mismatch");
    }
    if (output.size() != out_features) {
        throw std::invalid_argument("GEMV output size mismatch");
    }

    for (std::size_t row = 0; row < out_features; ++row) {
        float acc = bias.empty() ? 0.0F : bias[row];
        const std::size_t base = row * in_features;
        for (std::size_t col = 0; col < in_features; ++col) {
            acc += weight[base + col] * input[col];
        }
        output[row] = acc;
    }
}

}  // namespace pokitlms
