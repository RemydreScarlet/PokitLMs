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

}  // namespace pokitlms
