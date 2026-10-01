#pragma once

#include <cstddef>
#include <span>

namespace pokitlms {

void linear_f32(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::span<float> output,
    std::size_t in_features,
    std::size_t out_features);

}  // namespace pokitlms
