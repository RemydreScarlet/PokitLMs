#include "pokitlms/ops/causal_conv1d.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace pokitlms {
namespace {

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::invalid_argument("causal convolution dimensions overflow");
    }
    return lhs * rhs;
}

float silu(float value) noexcept {
    if (value >= 0.0F) return value / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return value * exponential / (1.0F + exponential);
}

}  // namespace

void causal_depthwise_conv1d_step(
    const float* input, const float* weights, std::size_t channels,
    std::size_t kernel_size, float* state, float* output) {
    if (!input || !weights || !output || channels == 0 || kernel_size == 0 ||
        (kernel_size > 1 && !state)) {
        throw std::invalid_argument("invalid causal convolution dimensions or buffers");
    }
    (void)checked_product(channels, kernel_size);
    const auto state_length = kernel_size - 1;
    (void)checked_product(channels, state_length);
    for (std::size_t channel = 0; channel < channels; ++channel) {
        const auto weight_offset = channel * kernel_size;
        const auto state_offset = channel * state_length;
        float sum = input[channel] * weights[weight_offset + state_length];
        for (std::size_t tap = 0; tap < state_length; ++tap) {
            sum += state[state_offset + tap] * weights[weight_offset + tap];
        }
        if (state_length != 0) {
            for (std::size_t tap = 0; tap + 1 < state_length; ++tap) {
                state[state_offset + tap] = state[state_offset + tap + 1];
            }
            state[state_offset + state_length - 1] = input[channel];
        }
        output[channel] = silu(sum);
    }
}

}  // namespace pokitlms
