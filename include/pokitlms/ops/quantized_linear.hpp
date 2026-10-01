#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pokitlms {

// Decodes one supported GGUF quantized row into FP32 values.
void dequantize_quantized_row(std::uint32_t ggml_type,
                              std::span<const std::byte> encoded,
                              std::span<float> output);

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

void linear_q2_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output);
void linear_q3_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output);
void linear_q4_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output);

}  // namespace pokitlms
