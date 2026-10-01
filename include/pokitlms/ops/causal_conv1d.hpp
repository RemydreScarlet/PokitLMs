#pragma once

#include <cstddef>

namespace pokitlms {

// Advances a depthwise causal Conv1d by one token. weights are laid out as
// [channels, kernel_size], state as [channels, kernel_size - 1] in oldest to
// newest order. The output uses SiLU, matching Qwen3.5's linear-attention path.
// state is updated in place; output may alias input.
void causal_depthwise_conv1d_step(
    const float* input, const float* weights, std::size_t channels,
    std::size_t kernel_size, float* state, float* output);

}  // namespace pokitlms
