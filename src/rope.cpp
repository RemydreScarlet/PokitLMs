#include "pokitlms/ops/rope.hpp"

#include <cmath>
#include <stdexcept>

namespace pokitlms {

namespace {

void rotate_heads(
    std::span<float> values,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t position,
    float theta) {
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t base = head * head_dim;
        for (std::size_t i = 0; i < head_dim; i += 2) {
            const float exponent =
                static_cast<float>(i) / static_cast<float>(head_dim);
            const float inv_freq = std::pow(theta, -exponent);
            const float angle = static_cast<float>(position) * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);

            const float x0 = values[base + i];
            const float x1 = values[base + i + 1];
            values[base + i] = x0 * c - x1 * s;
            values[base + i + 1] = x0 * s + x1 * c;
        }
    }
}

}  // namespace

void apply_rope(
    std::span<float> query,
    std::span<float> key,
    std::size_t num_heads,
    std::size_t num_kv_heads,
    std::size_t head_dim,
    std::size_t position,
    float theta) {
    if (head_dim == 0 || (head_dim % 2) != 0) {
        throw std::invalid_argument("RoPE head_dim must be a positive even number");
    }
    if (query.size() != num_heads * head_dim) {
        throw std::invalid_argument("RoPE query size mismatch");
    }
    if (key.size() != num_kv_heads * head_dim) {
        throw std::invalid_argument("RoPE key size mismatch");
    }
    if (theta <= 0.0F) {
        throw std::invalid_argument("RoPE theta must be positive");
    }

    rotate_heads(query, num_heads, head_dim, position, theta);
    rotate_heads(key, num_kv_heads, head_dim, position, theta);
}

}  // namespace pokitlms
