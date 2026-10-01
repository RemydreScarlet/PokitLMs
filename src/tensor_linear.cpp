#include "pokitlms/model/tensor_linear.hpp"

#include "pokitlms/ops/linear.hpp"
#include "pokitlms/ops/quantized_linear.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pokitlms::model {

void tensor_linear(const TensorReader& weights, std::span<const float> input,
                   std::span<float> output, std::size_t row_batch) {
    const auto& tensor = weights.tensor();
    if (tensor.dimensions.empty() || input.empty() || output.empty() || row_batch == 0 ||
        tensor.dimensions.front() != input.size() || weights.row_count() != output.size()) {
        throw std::invalid_argument("GGUF matrix dimensions do not match linear input/output");
    }

    const bool floating = tensor.type == 0 || tensor.type == 1;
    const bool quantized = tensor.type == 2 || tensor.type == 8 || tensor.type == 10 ||
        tensor.type == 11 || tensor.type == 12 || tensor.type == 13 || tensor.type == 14;
    if (!floating && !quantized) {
        throw std::invalid_argument("unsupported GGUF matrix type for tensor_linear");
    }

    const auto feature_count = input.size();
    const auto chunk_limit = std::min(row_batch, output.size());
    for (std::size_t first = 0; first < output.size();) {
        const auto rows = std::min(chunk_limit, output.size() - first);
        auto destination = output.subspan(first, rows);
        if (floating) {
            const auto values = weights.read_float_rows(first, rows);
            linear(input, values, {}, destination);
        } else {
            if (feature_count > std::numeric_limits<std::size_t>::max() / rows) {
                throw std::length_error("GGUF matrix batch dimensions overflow");
            }
            const auto encoded = weights.read_rows(first, rows);
            switch (tensor.type) {
                case 2: linear_q4_0(input, encoded, rows, {}, destination); break;
                case 8: linear_q8_0(input, encoded, rows, {}, destination); break;
                case 10: linear_q2_k(input, encoded, rows, {}, destination); break;
                case 11: linear_q3_k(input, encoded, rows, {}, destination); break;
                case 12: linear_q4_k(input, encoded, rows, {}, destination); break;
                case 13: linear_q5_k(input, encoded, rows, {}, destination); break;
                case 14: linear_q6_k(input, encoded, rows, {}, destination); break;
                default: throw std::logic_error("unreachable GGUF type dispatch");
            }
        }
        first += rows;
    }
}

}  // namespace pokitlms::model
