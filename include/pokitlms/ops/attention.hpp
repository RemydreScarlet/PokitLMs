#pragma once

#include <cstddef>
#include <span>

#include "pokitlms/kv_cache.hpp"

namespace pokitlms {

// Computes one decode-step attention output from a populated KV cache.
// Supports grouped-query attention when num_heads is divisible by num_kv_heads.
void attention_decode_f32(
    std::span<const float> query,
    const KVCache& cache,
    std::span<float> output,
    std::size_t num_heads,
    std::size_t num_kv_heads,
    std::size_t head_dim);

}  // namespace pokitlms
