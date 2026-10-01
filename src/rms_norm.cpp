#include "pokitlms/ops/rms_norm.hpp"

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

    float sum_squares = 0.0F;
    for (const float value : input) {
        sum_squares += value * value;
    }

    const float mean_square = sum_squares / static_cast<float>(input.size());
    const float scale = 1.0F / std::sqrt(mean_square + epsilon);

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}

}  // namespace pokitlms
