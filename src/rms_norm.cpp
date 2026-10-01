#include "pokitlms/ops/rms_norm.hpp"
#include "simd_kernels.hpp"

#include <cmath>
#include <stdexcept>

namespace pokitlms {

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    float epsilon) {
    if (input.size() != weight.size() || input.size() != output.size()) {
        throw std::invalid_argument("RMSNorm spans must have equal sizes");
    }
    if (input.empty()) {
        return;
    }

    const float sum_squares = detail::dot_f32(input.data(), input.data(), input.size());

    const float mean_square = sum_squares / static_cast<float>(input.size());
    const float scale = 1.0F / std::sqrt(mean_square + epsilon);

    detail::rms_scale_f32(input.data(), weight.data(), output.data(), scale, input.size());
}

}  // namespace pokitlms
