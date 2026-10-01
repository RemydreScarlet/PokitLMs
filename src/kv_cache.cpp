#include "pokitlms/kv_cache.hpp"

#include <stdexcept>

namespace pokitlms {

KVCache::KVCache(
    std::size_t max_seq_len,
    std::size_t num_kv_heads,
    std::size_t head_dim)
    : max_seq_len_(max_seq_len),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      keys_(max_seq_len * num_kv_heads * head_dim),
      values_(max_seq_len * num_kv_heads * head_dim) {
    if (max_seq_len == 0 || num_kv_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("KVCache dimensions must be non-zero");
    }
}

void KVCache::clear() noexcept {
    size_ = 0;
}

void KVCache::append(
    std::span<const float> key,
    std::span<const float> value) {
    const std::size_t token_width = num_kv_heads_ * head_dim_;
    if (key.size() != token_width || value.size() != token_width) {
        throw std::invalid_argument("KVCache token size mismatch");
    }
    if (size_ >= max_seq_len_) {
        throw std::out_of_range("KVCache capacity exceeded");
    }

    const std::size_t offset = size_ * token_width;
    for (std::size_t i = 0; i < token_width; ++i) {
        keys_[offset + i] = key[i];
        values_[offset + i] = value[i];
    }
    ++size_;
}

std::size_t KVCache::size() const noexcept {
    return size_;
}

std::size_t KVCache::capacity() const noexcept {
    return max_seq_len_;
}

std::size_t KVCache::num_kv_heads() const noexcept {
    return num_kv_heads_;
}

std::size_t KVCache::head_dim() const noexcept {
    return head_dim_;
}

std::span<const float> KVCache::key_at(std::size_t position) const {
    if (position >= size_) {
        throw std::out_of_range("KVCache key position out of range");
    }
    const std::size_t width = num_kv_heads_ * head_dim_;
    return std::span<const float>(keys_.data() + position * width, width);
}

std::span<const float> KVCache::value_at(std::size_t position) const {
    if (position >= size_) {
        throw std::out_of_range("KVCache value position out of range");
    }
    const std::size_t width = num_kv_heads_ * head_dim_;
    return std::span<const float>(values_.data() + position * width, width);
}

}  // namespace pokitlms
