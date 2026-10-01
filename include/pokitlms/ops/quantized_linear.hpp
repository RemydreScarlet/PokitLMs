#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace pokitlms {

// Scalar reference matmuls for GGUF Q4_0 (type 2) and Q8_0 (type 8).
// Weight rows are contiguous and laid out as [output_features, input_features].
void linear_q4_0(std::span<const float> input,
                 std::span<const std::byte> weights,
                 std::size_t output_features,
                 std::span<const float> bias,
                 std::span<float> output);

void linear_q8_0(std::span<const float> input,
                 std::span<const std::byte> weights,
                 std::size_t output_features,
                 std::span<const float> bias,
                 std::span<float> output);

}  // namespace pokitlms
