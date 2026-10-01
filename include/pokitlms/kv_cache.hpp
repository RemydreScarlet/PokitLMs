#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace pokitlms {

class KVCache {
public:
    KVCache(
        std::size_t max_seq_len,
        std::size_t num_kv_heads,
        std::size_t head_dim);

    void clear() noexcept;

    void append(
        std::span<const float> key,
        std::span<const float> value);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t num_kv_heads() const noexcept;
    [[nodiscard]] std::size_t head_dim() const noexcept;

    [[nodiscard]] std::span<const float> key_at(std::size_t position) const;
    [[nodiscard]] std::span<const float> value_at(std::size_t position) const;

private:
    std::size_t max_seq_len_;
    std::size_t num_kv_heads_;
    std::size_t head_dim_;
    std::size_t size_ = 0;
    std::vector<float> keys_;
    std::vector<float> values_;
};

}  // namespace pokitlms
