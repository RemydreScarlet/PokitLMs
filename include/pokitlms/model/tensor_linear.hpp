#pragma once

#include "pokitlms/model/tensor_reader.hpp"

#include <cstddef>
#include <span>

namespace pokitlms::model {

// Applies one GGUF matrix [input_features, output_features] to an input vector.
// Weight rows are fetched in bounded batches so the full matrix stays on disk.
// row_batch=0 selects a batch sized to approximately 256 KiB of temporary weights.
// Supported tensor types: F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0.
void tensor_linear(const TensorReader& weights, std::span<const float> input,
                   std::span<float> output, std::size_t row_batch = 0);

}  // namespace pokitlms::model
