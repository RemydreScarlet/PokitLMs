#include "pokitlms/ops/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pokitlms {
namespace {

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::invalid_argument("KV cache dimensions overflow");
    }
    return lhs * rhs;
}

}  // namespace

KvCache::KvCache(std::size_t capacity, std::size_t kv_heads,
                 std::size_t key_dimension, std::size_t value_dimension)
    : capacity_(capacity), kv_heads_(kv_heads), key_dimension_(key_dimension),
      value_dimension_(value_dimension),
      keys_(checked_product(checked_product(capacity, kv_heads), key_dimension)),
      values_(checked_product(checked_product(capacity, kv_heads), value_dimension)),
      scores_(capacity) {
    if (capacity == 0 || kv_heads == 0 || key_dimension == 0 || value_dimension == 0) {
        throw std::invalid_argument("KV cache dimensions must be nonzero");
    }
}

void KvCache::append(std::uint64_t position, const float* keys, std::size_t key_count,
                     const float* values, std::size_t value_count) {
    const auto expected_keys = checked_product(kv_heads_, key_dimension_);
    const auto expected_values = checked_product(kv_heads_, value_dimension_);
    if (!keys || !values || key_count != expected_keys || value_count != expected_values) {
        throw std::invalid_argument("KV cache append has invalid data dimensions");
    }
    if (has_position_ && (last_position_ == std::numeric_limits<std::uint64_t>::max() ||
                          position != last_position_ + 1)) {
        throw std::invalid_argument("KV cache positions must be consecutive");
    }

    if (!has_position_) {
        first_position_ = position;
        last_position_ = position;
        has_position_ = true;
        size_ = 1;
    } else {
        last_position_ = position;
        if (size_ < capacity_) ++size_;
        else ++first_position_;
    }

    const auto slot = static_cast<std::size_t>(position % capacity_);
    std::copy_n(keys, expected_keys, keys_.data() + slot * expected_keys);
    std::copy_n(values, expected_values, values_.data() + slot * expected_values);
}

void KvCache::attend(const float* query, std::size_t query_count, float* output,
                     std::size_t output_count, float scale) {
    if (!has_position_) throw std::logic_error("cannot attend with an empty KV cache");
    const auto key_size = checked_product(kv_heads_, key_dimension_);
    const auto value_size = checked_product(kv_heads_, value_dimension_);
    if (!query || !output || query_count == 0 || query_count % key_dimension_ != 0 ||
        output_count != checked_product(query_count / key_dimension_, value_dimension_)) {
        throw std::invalid_argument("attention query/output dimensions do not match");
    }
    const auto query_heads = query_count / key_dimension_;
    if (query_heads % kv_heads_ != 0) {
        throw std::invalid_argument("query heads must be evenly grouped over KV heads");
    }
    if (scale == 0.0F) scale = 1.0F / std::sqrt(static_cast<float>(key_dimension_));
    if (!std::isfinite(scale) || scale <= 0.0F) throw std::invalid_argument("attention scale must be positive");

    const auto group_size = query_heads / kv_heads_;
    for (std::size_t head = 0; head < query_heads; ++head) {
        const auto kv_head = head / group_size;
        const auto* q = query + head * key_dimension_;
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t index = 0; index < size_; ++index) {
            const auto position = first_position_ + index;
            const auto slot = static_cast<std::size_t>(position % capacity_);
            const auto* key = keys_.data() + slot * key_size + kv_head * key_dimension_;
            float score = 0.0F;
            for (std::size_t d = 0; d < key_dimension_; ++d) score += q[d] * key[d];
            score *= scale;
            scores_[index] = score;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0F;
        for (std::size_t index = 0; index < size_; ++index) {
            scores_[index] = std::exp(scores_[index] - max_score);
            denominator += scores_[index];
        }
        auto* out = output + head * value_dimension_;
        std::fill_n(out, value_dimension_, 0.0F);
        for (std::size_t index = 0; index < size_; ++index) {
            const auto position = first_position_ + index;
            const auto slot = static_cast<std::size_t>(position % capacity_);
            const auto* value = values_.data() + slot * value_size + kv_head * value_dimension_;
            const float probability = scores_[index] / denominator;
            for (std::size_t d = 0; d < value_dimension_; ++d) out[d] += probability * value[d];
        }
    }
}

std::size_t KvCache::size() const noexcept { return size_; }
std::size_t KvCache::capacity() const noexcept { return capacity_; }
std::uint64_t KvCache::first_position() const noexcept { return first_position_; }
std::uint64_t KvCache::last_position() const noexcept { return last_position_; }
std::size_t KvCache::storage_bytes() const noexcept {
    return (keys_.size() + values_.size()) * sizeof(float);
}
void KvCache::clear() noexcept {
    size_ = 0;
    has_position_ = false;
    first_position_ = 0;
    last_position_ = 0;
}

}  // namespace pokitlms
