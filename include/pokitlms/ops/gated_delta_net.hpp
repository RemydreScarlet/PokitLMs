#pragma once

#include <cstddef>
#include <vector>

namespace pokitlms {

// Reusable scratch for one-token recurrent gated-delta attention.
struct GatedDeltaNetScratch {
    std::vector<float> normalized_query;
    std::vector<float> normalized_key;
    std::vector<float> delta;
};

// Advances one recurrent step. query/key are [heads, key_dimension], value and
// output are [heads, value_dimension], and state is [heads, key_dimension,
// value_dimension]. log_decay is the non-positive per-head log decay and beta
// is the per-head update gate in [0, 1]. Query/key are L2-normalized with eps
// 1e-6; query is additionally scaled by 1/sqrt(key_dimension).
void gated_delta_recurrent_step(
    const float* query, const float* key, const float* value,
    const float* log_decay, const float* beta,
    std::size_t heads, std::size_t key_dimension, std::size_t value_dimension,
    float* state, float* output, GatedDeltaNetScratch& scratch);

// Qwen3.5 may use fewer key/query heads than value heads. Each value head
// shares one key/query head with its contiguous repeat group; recurrent state
// and decay/update gates remain per value head.
void gated_delta_recurrent_step_grouped(
    const float* query, const float* key, const float* value,
    const float* log_decay, const float* beta,
    std::size_t key_heads, std::size_t value_heads,
    std::size_t key_dimension, std::size_t value_dimension,
    float* state, float* output, GatedDeltaNetScratch& scratch);

}  // namespace pokitlms
