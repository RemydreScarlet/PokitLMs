#include "pokitlms/ops/feed_forward.hpp"

#include "pokitlms/ops/linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pokitlms {
namespace {

std::size_t matrix_size(std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::invalid_argument("feed-forward dimensions overflow");
    }
    return rows * columns;
}

std::size_t intermediate_size(
    std::size_t input_size,
    const SwiGluWeights& weights,
    std::size_t output_size) {
    if (input_size == 0 || output_size != input_size || weights.gate.size() % input_size != 0) {
        throw std::invalid_argument("invalid SwiGLU dimensions");
    }
    const auto intermediate = weights.gate.size() / input_size;
    if (intermediate == 0 || weights.up.size() != weights.gate.size() ||
        weights.down.size() != matrix_size(input_size, intermediate)) {
        throw std::invalid_argument("SwiGLU weight sizes do not match dimensions");
    }
    return intermediate;
}

void run_expert(
    std::span<const float> input,
    const SwiGluWeights& weights,
    std::span<float> output) {
    const auto intermediate = intermediate_size(input.size(), weights, output.size());
    std::vector<float> gate(intermediate);
    std::vector<float> up(intermediate);
    for (std::size_t row = 0; row < intermediate; ++row) {
        const auto offset = row * input.size();
        float gate_value = 0.0F;
        float up_value = 0.0F;
        for (std::size_t column = 0; column < input.size(); ++column) {
            gate_value += weights.gate[offset + column] * input[column];
            up_value += weights.up[offset + column] * input[column];
        }
        gate[row] = (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
    }
    linear(gate, weights.down, {}, output);
}

}  // namespace

void swi_glu(
    std::span<const float> input,
    const SwiGluWeights& weights,
    std::span<float> output) {
    run_expert(input, weights, output);
}

void moe_swi_glu(
    std::span<const float> input,
    std::span<const float> router_weights,
    const std::vector<MoeExpert>& experts,
    std::size_t top_k,
    std::span<float> output) {
    if (input.empty() || output.size() != input.size() || experts.empty() ||
        top_k == 0 || top_k > experts.size() ||
        router_weights.size() != matrix_size(experts.size(), input.size())) {
        throw std::invalid_argument("invalid MoE dimensions or top_k");
    }

    std::vector<float> logits(experts.size());
    for (std::size_t expert = 0; expert < experts.size(); ++expert) {
        const auto offset = expert * input.size();
        for (std::size_t column = 0; column < input.size(); ++column) {
            logits[expert] += router_weights[offset + column] * input[column];
        }
    }

    std::vector<std::size_t> selected(experts.size());
    for (std::size_t i = 0; i < selected.size(); ++i) selected[i] = i;
    std::partial_sort(selected.begin(), selected.begin() + static_cast<std::ptrdiff_t>(top_k),
                      selected.end(), [&logits](std::size_t lhs, std::size_t rhs) {
                          return logits[lhs] > logits[rhs];
                      });
    selected.resize(top_k);

    const float max_logit = logits[selected.front()];
    float normalization = 0.0F;
    std::vector<float> route_weights(top_k);
    for (std::size_t i = 0; i < top_k; ++i) {
        route_weights[i] = std::exp(logits[selected[i]] - max_logit);
        normalization += route_weights[i];
    }

    std::vector<float> result(output.size(), 0.0F);
    std::vector<float> expert_output(output.size());
    for (std::size_t i = 0; i < top_k; ++i) {
        run_expert(input, experts[selected[i]].weights, expert_output);
        const float scale = route_weights[i] / normalization;
        for (std::size_t j = 0; j < output.size(); ++j) {
            result[j] += scale * expert_output[j];
        }
    }
    std::copy(result.begin(), result.end(), output.begin());
}

}  // namespace pokitlms
