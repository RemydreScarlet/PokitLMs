#pragma once

#include <cstddef>
#include <span>

namespace pokitlms {

void apply_rope(
    std::span<float> query,
    std::span<float> key,
    std::size_t num_heads,
    std::size_t num_kv_heads,
    std::size_t head_dim,
    std::size_t position,
    float theta = 10000.0F);

}  // namespace pokitlms
