#include "pokitlms/ops/gated_delta_net.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace pokitlms {
namespace {

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::invalid_argument("gated-delta dimensions overflow");
    }
    return lhs * rhs;
}

}  // namespace

void gated_delta_recurrent_step(
    const float* query, const float* key, const float* value,
    const float* log_decay, const float* beta,
    std::size_t heads, std::size_t key_dimension, std::size_t value_dimension,
    float* state, float* output, GatedDeltaNetScratch& scratch) {
    if (!query || !key || !value || !log_decay || !beta || !state || !output ||
        heads == 0 || key_dimension == 0 || value_dimension == 0) {
        throw std::invalid_argument("invalid gated-delta step dimensions or buffers");
    }
    const auto qk_count = checked_product(heads, key_dimension);
    const auto value_count = checked_product(heads, value_dimension);
    const auto state_head_size = checked_product(key_dimension, value_dimension);
    const auto state_count = checked_product(heads, state_head_size);
    (void)value_count;
    (void)state_count;
    for (std::size_t head = 0; head < heads; ++head) {
        if (!std::isfinite(log_decay[head]) || log_decay[head] > 0.0F ||
            !std::isfinite(beta[head]) || beta[head] < 0.0F || beta[head] > 1.0F) {
            throw std::invalid_argument("invalid gated-delta decay or update gate");
        }
    }
    scratch.normalized_query.resize(qk_count);
    scratch.normalized_key.resize(qk_count);
    scratch.delta.resize(value_count);

    constexpr float norm_epsilon = 1.0e-6F;
    const float query_scale = 1.0F / std::sqrt(static_cast<float>(key_dimension));
    for (std::size_t head = 0; head < heads; ++head) {
        const auto q_offset = head * key_dimension;
        float query_square_sum = 0.0F;
        float key_square_sum = 0.0F;
        for (std::size_t i = 0; i < key_dimension; ++i) {
            query_square_sum += query[q_offset + i] * query[q_offset + i];
            key_square_sum += key[q_offset + i] * key[q_offset + i];
        }
        const float query_inverse_norm = 1.0F / std::sqrt(query_square_sum + norm_epsilon);
        const float key_inverse_norm = 1.0F / std::sqrt(key_square_sum + norm_epsilon);
        for (std::size_t i = 0; i < key_dimension; ++i) {
            scratch.normalized_query[q_offset + i] =
                query[q_offset + i] * query_inverse_norm * query_scale;
            scratch.normalized_key[q_offset + i] = key[q_offset + i] * key_inverse_norm;
        }

        const auto state_offset = head * state_head_size;
        const float decay = std::exp(log_decay[head]);
        for (std::size_t i = 0; i < state_head_size; ++i) {
            state[state_offset + i] *= decay;
        }

        const auto delta_offset = head * value_dimension;
        for (std::size_t v = 0; v < value_dimension; ++v) {
            float memory = 0.0F;
            for (std::size_t k = 0; k < key_dimension; ++k) {
                memory += state[state_offset + k * value_dimension + v] *
                          scratch.normalized_key[q_offset + k];
            }
            scratch.delta[delta_offset + v] = (value[delta_offset + v] - memory) * beta[head];
            output[delta_offset + v] = 0.0F;
        }

        for (std::size_t k = 0; k < key_dimension; ++k) {
            const float normalized_key = scratch.normalized_key[q_offset + k];
            const float normalized_query = scratch.normalized_query[q_offset + k];
            const auto row = state_offset + k * value_dimension;
            for (std::size_t v = 0; v < value_dimension; ++v) {
                auto& cell = state[row + v];
                cell += normalized_key * scratch.delta[delta_offset + v];
                output[delta_offset + v] += cell * normalized_query;
            }
        }
    }
}

}  // namespace pokitlms
