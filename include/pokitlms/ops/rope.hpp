#pragma once

#include <cstddef>
#include <span>

namespace pokitlms {

// Applies rotary position embeddings in place to contiguous heads. Each head
// has head_dimension values; rotary_dimension must be even and no larger than
// head_dimension. Values after rotary_dimension are left unchanged.
void rope_inplace(
    std::span<float> values,
    std::size_t head_dimension,
    std::size_t rotary_dimension,
    std::size_t position,
    float theta = 10000.0F);

// Applies full-head rotary embeddings to query and key tensors.
void apply_rope(std::span<float> query, std::span<float> key,
                std::size_t num_heads, std::size_t num_kv_heads,
                std::size_t head_dim, std::size_t position,
                float theta = 10000.0F);

}  // namespace pokitlms
