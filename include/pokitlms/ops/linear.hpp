#pragma once

#include <span>

namespace pokitlms {

// Computes output = weight * input + bias, with weight stored row-major
// as [output_features, input_features]. An empty bias omits the bias term.
void linear(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::span<float> output);

}  // namespace pokitlms
