#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace pokitlms {

struct SwiGluWeights {
    // gate and up: [intermediate_features, input_features]
    // down: [input_features, intermediate_features]
    std::span<const float> gate;
    std::span<const float> up;
    std::span<const float> down;
};

struct MoeExpert {
    SwiGluWeights weights;
};

using MoeExpertRunner = std::function<void(
    std::size_t expert_id, std::span<const float> input, std::span<float> output)>;

// Computes down(silu(gate(input)) * up(input)).
void swi_glu(
    std::span<const float> input,
    const SwiGluWeights& weights,
    std::span<float> output);

// Routes input to the top_k experts. Router weights are row-major
// [expert_count, input_features]; selected router logits are softmaxed and
// used to combine expert outputs.
void moe_swi_glu(
    std::span<const float> input,
    std::span<const float> router_weights,
    const std::vector<MoeExpert>& experts,
    std::size_t top_k,
    std::span<float> output);

// Runs only the selected experts. The callback can fetch the routed expert
// from bounded storage, execute it, and release it before the next callback.
void moe_swi_glu_streaming(
    std::span<const float> input,
    std::span<const float> router_weights,
    std::size_t expert_count,
    std::size_t top_k,
    const MoeExpertRunner& runner,
    std::span<float> output);

}  // namespace pokitlms
