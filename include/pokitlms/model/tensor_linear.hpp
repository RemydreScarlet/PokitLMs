#pragma once

#include "pokitlms/model/tensor_reader.hpp"

#include <cstddef>
#include <span>

namespace pokitlms::model {

// Applies one GGUF matrix [input_features, output_features] to an input vector.
// Weight rows are fetched in bounded batches so the full matrix stays on disk.
// Supported tensor types: F32, F16, Q2_K, Q3_K, Q4_0, Q4_K, and Q8_0.
void tensor_linear(const TensorReader& weights, std::span<const float> input,
                   std::span<float> output, std::size_t row_batch = 16);

}  // namespace pokitlms::model
