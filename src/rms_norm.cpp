#include "pokitlms/ops/rms_norm.hpp"
#include "simd_kernels.hpp"

#include <cmath>
#include <stdexcept>

namespace pokitlms {

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    float epsilon) {
    if (input.size() != weight.size() || input.size() != output.size()) {
        throw std::invalid_argument("RMSNorm spans must have equal sizes");
    }
    if (input.empty()) {
        return;
    }

    const float sum_squares = detail::dot_f32(input.data(), input.data(), input.size());

    const float mean_square = sum_squares / static_cast<float>(input.size());
    const float scale = 1.0F / std::sqrt(mean_square + epsilon);

    detail::rms_scale_f32(input.data(), weight.data(), output.data(), scale, input.size());
}

void rms_norm_gated(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> gate,
    std::span<float> output,
    std::size_t rows,
    float epsilon) {
    if (weight.empty() || input.size() != gate.size() || input.size() != output.size() ||
        input.size() % weight.size() != 0 || input.size() / weight.size() != rows ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument("gated RMSNorm dimensions or epsilon are invalid");
    }
    const auto width = weight.size();
    for (std::size_t row = 0; row < rows; ++row) {
        const auto offset = row * width;
        const auto values = input.subspan(offset, width);
        const float mean_square = detail::dot_f32(values.data(), values.data(), width) /
                                  static_cast<float>(width);
        const float scale = 1.0F / std::sqrt(mean_square + epsilon);
        for (std::size_t i = 0; i < width; ++i) {
            const float gate_value = gate[offset + i];
            const float sigmoid = gate_value >= 0.0F
                ? 1.0F / (1.0F + std::exp(-gate_value))
                : std::exp(gate_value) / (1.0F + std::exp(gate_value));
            output[offset + i] = input[offset + i] * scale * weight[i] *
                                 (gate_value * sigmoid);
        }
    }
}

}  // namespace pokitlms
