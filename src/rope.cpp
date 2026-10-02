#include "pokitlms/ops/rope.hpp"

#include <cmath>
#include <stdexcept>

namespace pokitlms {

void rope_inplace(
    std::span<float> values,
    std::size_t head_dimension,
    std::size_t rotary_dimension,
    std::size_t position,
    float theta) {
    if (head_dimension == 0 || values.size() % head_dimension != 0 ||
        rotary_dimension == 0 || rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 || !(theta > 0.0F) || !std::isfinite(theta)) {
        throw std::invalid_argument("invalid RoPE dimensions or theta");
    }

    const float position_value = static_cast<float>(position);
    const auto head_count = values.size() / head_dimension;
    for (std::size_t head = 0; head < head_count; ++head) {
        const auto head_offset = head * head_dimension;
        for (std::size_t pair = 0; pair < rotary_dimension / 2; ++pair) {
            const float exponent = -2.0F * static_cast<float>(pair) /
                                   static_cast<float>(rotary_dimension);
            const float angle = position_value * std::pow(theta, exponent);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const auto index = head_offset + pair * 2;
            const float even = values[index];
            const float odd = values[index + 1];
            values[index] = even * cosine - odd * sine;
            values[index + 1] = even * sine + odd * cosine;
        }
    }
}

void apply_rope(std::span<float> query, std::span<float> key,
                std::size_t num_heads, std::size_t num_kv_heads,
                std::size_t head_dim, std::size_t position, float theta) {
    if (head_dim == 0 || query.size() % head_dim != 0 || key.size() % head_dim != 0 ||
        query.size() / head_dim != num_heads || key.size() / head_dim != num_kv_heads) {
        throw std::invalid_argument("RoPE query/key size mismatch");
    }
    rope_inplace(query, head_dim, head_dim, position, theta);
    rope_inplace(key, head_dim, head_dim, position, theta);
}

}  // namespace pokitlms
