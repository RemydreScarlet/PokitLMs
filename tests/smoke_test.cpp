#include "pokitlms/ops/rms_norm.hpp"
#include "pokitlms/pokitlms.h"
#include "pokitlms/tensor.hpp"

#include <cassert>
#include <cmath>
#include <string_view>

int main() {
    using pokitlms::Tensor;

    Tensor tensor({4});
    assert(tensor.numel() == 4);
    assert(std::string_view(pokitlms_version()) == "0.1.0");

    const float input[] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {};

    pokitlms::rms_norm(input, weight, output);

    const float mean_square = (1.0F + 4.0F + 9.0F + 16.0F) / 4.0F;
    const float expected0 = 1.0F / std::sqrt(mean_square + 1.0e-5F);
    assert(std::fabs(output[0] - expected0) < 1.0e-5F);

    return 0;
}
