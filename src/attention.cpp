#include "pokitlms/ops/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pokitlms {

void attention_decode_f32(
    std::span<const float> query,
    const KVCache& cache,
    std::span<float> output,
    std::size_t num_heads,
    std::size_t num_kv_heads,
    std::size_t head_dim) {
    if (num_heads == 0 || num_kv_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("attention dimensions must be non-zero");
    }
    if ((num_heads % num_kv_heads) != 0) {
        throw std::invalid_argument("num_heads must be divisible by num_kv_heads");
    }
    if (query.size() != num_heads * head_dim ||
        output.size() != num_heads * head_dim) {
        throw std::invalid_argument("attention query/output size mismatch");
    }
    if (cache.num_kv_heads() != num_kv_heads ||
        cache.head_dim() != head_dim) {
        throw std::invalid_argument("attention cache shape mismatch");
    }
    if (cache.size() == 0) {
        throw std::invalid_argument("attention requires a non-empty KV cache");
    }

    std::fill(output.begin(), output.end(), 0.0F);

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const std::size_t heads_per_kv = num_heads / num_kv_heads;
    std::vector<float> scores(cache.size());

    for (std::size_t head = 0; head < num_heads; ++head) {
        const std::size_t kv_head = head / heads_per_kv;
        const std::size_t q_base = head * head_dim;
        float max_score = -std::numeric_limits<float>::infinity();

        for (std::size_t pos = 0; pos < cache.size(); ++pos) {
            const auto key = cache.key_at(pos);
            const std::size_t k_base = kv_head * head_dim;
            float dot = 0.0F;
            for (std::size_t d = 0; d < head_dim; ++d) {
                dot += query[q_base + d] * key[k_base + d];
            }
            scores[pos] = dot * scale;
            max_score = std::max(max_score, scores[pos]);
        }

        float denom = 0.0F;
        for (float& score : scores) {
            score = std::exp(score - max_score);
            denom += score;
        }

        const std::size_t out_base = head * head_dim;
        for (std::size_t pos = 0; pos < cache.size(); ++pos) {
            const auto value = cache.value_at(pos);
            const std::size_t v_base = kv_head * head_dim;
            const float probability = scores[pos] / denom;
            for (std::size_t d = 0; d < head_dim; ++d) {
                output[out_base + d] += probability * value[v_base + d];
            }
        }
    }
}

}  // namespace pokitlms
