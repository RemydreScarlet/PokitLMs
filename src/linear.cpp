#include "pokitlms/ops/linear.hpp"
#include "simd_kernels.hpp"

#include <limits>
#include <stdexcept>

namespace pokitlms {

void linear(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::span<float> output) {
    if (input.empty()) {
        throw std::invalid_argument("Linear input must not be empty");
    }
    if (output.size() > std::numeric_limits<std::size_t>::max() / input.size() ||
        weight.size() != input.size() * output.size()) {
        throw std::invalid_argument("Linear weight size must equal input_features * output_features");
    }
    if (!bias.empty() && bias.size() != output.size()) {
        throw std::invalid_argument("Linear bias size must equal output_features");
    }

    for (std::size_t row = 0; row < output.size(); ++row) {
        const auto row_offset = row * input.size();
        output[row] = (bias.empty() ? 0.0F : bias[row]) +
            detail::dot_f32(weight.data() + row_offset, input.data(), input.size());
    }
}

}  // namespace pokitlms
