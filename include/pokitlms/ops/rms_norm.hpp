#pragma once

#include <span>

namespace pokitlms {

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    float epsilon = 1.0e-5F);

// Applies row-wise RMSNorm, learned weights, then SiLU(gate), matching the
// gated normalization used by Qwen3.5 Gated DeltaNet.
void rms_norm_gated(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> gate,
    std::span<float> output,
    std::size_t rows,
    float epsilon = 1.0e-6F);

// RMSNorm variant whose stored weights are centered at zero and applied as
// (1 + weight), matching Qwen3.5's standard attention and decoder norms.
void rms_norm_zero_centered(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    float epsilon = 1.0e-6F);

}  // namespace pokitlms
