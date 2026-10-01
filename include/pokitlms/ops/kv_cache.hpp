#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pokitlms {

enum class KvCachePrecision : std::uint8_t { Float32, Float16, Q8_0 };

// Fixed-capacity, single-sequence KV state for autoregressive decode.
// Storage is [position, KV head, head dimension]; old positions are evicted
// in a ring when capacity is reached.
class KvCache {
public:
    KvCache(std::size_t capacity, std::size_t kv_heads,
            std::size_t key_dimension, std::size_t value_dimension,
            KvCachePrecision precision = KvCachePrecision::Float32);

    void append(std::uint64_t position, const float* keys, std::size_t key_count,
                const float* values, std::size_t value_count);

    // query is [query_heads, key_dimension], output is
    // [query_heads, value_dimension]. Query heads are grouped evenly over KV heads.
    void attend(const float* query, std::size_t query_count, float* output,
                std::size_t output_count, float scale = 0.0F);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::uint64_t first_position() const noexcept;
    [[nodiscard]] std::uint64_t last_position() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    void clear() noexcept;

private:
    std::size_t capacity_;
    KvCachePrecision precision_;
    std::size_t kv_heads_;
    std::size_t key_dimension_;
    std::size_t value_dimension_;
    std::size_t size_{};
    std::uint64_t first_position_{};
    std::uint64_t last_position_{};
    bool has_position_{};
    std::vector<float> keys_;
    std::vector<float> values_;
    std::vector<std::uint16_t> keys_f16_;
    std::vector<std::uint16_t> values_f16_;
    std::vector<std::int8_t> keys_q8_;
    std::vector<std::int8_t> values_q8_;
    std::vector<std::uint16_t> keys_q8_scales_;
    std::vector<std::uint16_t> values_q8_scales_;
    std::size_t key_blocks_per_head_{};
    std::size_t value_blocks_per_head_{};
    std::vector<float> scores_;
};

}  // namespace pokitlms
