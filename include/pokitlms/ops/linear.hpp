#pragma once

#include <cstddef>
#include <span>

namespace pokitlms {

// Computes output = weight * input + bias, with weight stored row-major
// as [output_features, input_features]. An empty bias omits the bias term.
void linear(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::span<float> output);

// Reference API with explicit matrix dimensions.
void linear_f32(std::span<const float> input, std::span<const float> weight,
                std::span<const float> bias, std::span<float> output,
                std::size_t in_features, std::size_t out_features);
void gemv_f32(std::span<const float> input, std::span<const float> weight,
              std::span<const float> bias, std::span<float> output,
              std::size_t in_features, std::size_t out_features);

}  // namespace pokitlms
