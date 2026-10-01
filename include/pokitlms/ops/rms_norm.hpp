#pragma once

#include <span>

namespace pokitlms {

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    float epsilon = 1.0e-5F);

}  // namespace pokitlms
