#include "pokitlms/ops/kv_cache.hpp"
#include "simd_kernels.hpp"

#include <bit>
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

std::uint16_t float_to_half(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const std::uint32_t exponent = (bits >> 23) & 0xffU;
    std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa ? 0x0200U : 0U));
    }
    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000U;
        const auto shift = static_cast<unsigned>(14 - half_exponent);
        const std::uint32_t rounding = (1U << (shift - 1)) - 1U + ((mantissa >> shift) & 1U);
        return static_cast<std::uint16_t>(sign | ((mantissa + rounding) >> shift));
    }
    mantissa += 0x0fffU + ((mantissa >> 13) & 1U);
    if (mantissa & 0x800000U) {
        mantissa = 0;
        if (++half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint16_t>(half_exponent) << 10) |
                                      static_cast<std::uint16_t>(mantissa >> 13));
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fU;
    std::uint32_t mantissa = bits & 0x03ffU;
    std::uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) return std::bit_cast<float>(sign);
        int unbiased = -14;
        while ((mantissa & 0x0400U) == 0) { mantissa <<= 1; --unbiased; }
        mantissa &= 0x03ffU;
        result = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23) | (mantissa << 13);
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(result);
}

void quantize_q8_block(const float* source, std::int8_t* destination,
                       std::uint16_t* scale_storage) {
    float maximum = 0.0F;
    for (std::size_t i = 0; i < 32; ++i) {
        if (!std::isfinite(source[i])) throw std::runtime_error("non-finite value in Q8 KV cache");
        maximum = std::max(maximum, std::abs(source[i]));
    }
    const float scale = maximum / 127.0F;
    *scale_storage = float_to_half(scale);
    const double inverse_scale = scale == 0.0F ? 0.0 : 1.0 / static_cast<double>(scale);
    for (std::size_t i = 0; i < 32; ++i) {
        const auto quantized = static_cast<int>(std::round(
            static_cast<double>(source[i]) * inverse_scale));
        destination[i] = static_cast<std::int8_t>(std::clamp(quantized, -127, 127));
    }
}

}  // namespace

KvCache::KvCache(std::size_t capacity, std::size_t kv_heads,
                 std::size_t key_dimension, std::size_t value_dimension,
                 KvCachePrecision precision)
    : capacity_(capacity), precision_(precision), kv_heads_(kv_heads), key_dimension_(key_dimension),
      value_dimension_(value_dimension),
      keys_(precision == KvCachePrecision::Float32
                ? checked_product(checked_product(capacity, kv_heads), key_dimension) : 0),
      values_(precision == KvCachePrecision::Float32
                  ? checked_product(checked_product(capacity, kv_heads), value_dimension) : 0),
      keys_f16_(precision == KvCachePrecision::Float16
                    ? checked_product(checked_product(capacity, kv_heads), key_dimension) : 0),
      values_f16_(precision == KvCachePrecision::Float16
                      ? checked_product(checked_product(capacity, kv_heads), value_dimension) : 0),
      keys_q8_(precision == KvCachePrecision::Q8_0
                   ? checked_product(checked_product(capacity, kv_heads), key_dimension) : 0),
      values_q8_(precision == KvCachePrecision::Q8_0
                     ? checked_product(checked_product(capacity, kv_heads), value_dimension) : 0),
      keys_q8_scales_(precision == KvCachePrecision::Q8_0
                          ? checked_product(checked_product(capacity, kv_heads), key_dimension / 32) : 0),
      values_q8_scales_(precision == KvCachePrecision::Q8_0
                            ? checked_product(checked_product(capacity, kv_heads), value_dimension / 32) : 0),
      key_blocks_per_head_(key_dimension / 32), value_blocks_per_head_(value_dimension / 32),
      scores_(capacity) {
    if (capacity == 0 || kv_heads == 0 || key_dimension == 0 || value_dimension == 0) {
        throw std::invalid_argument("KV cache dimensions must be nonzero");
    }
    if (precision_ == KvCachePrecision::Q8_0 &&
        (key_dimension_ % 32 != 0 || value_dimension_ % 32 != 0)) {
        throw std::invalid_argument("Q8_0 KV cache dimensions must be multiples of 32");
    }
    if (precision_ != KvCachePrecision::Float32 && precision_ != KvCachePrecision::Float16 &&
        precision_ != KvCachePrecision::Q8_0) {
        throw std::invalid_argument("unsupported KV cache precision");
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
    const auto key_offset = slot * expected_keys;
    const auto value_offset = slot * expected_values;
    if (precision_ == KvCachePrecision::Float32) {
        std::copy_n(keys, expected_keys, keys_.data() + key_offset);
        std::copy_n(values, expected_values, values_.data() + value_offset);
    } else if (precision_ == KvCachePrecision::Float16) {
        for (std::size_t i = 0; i < expected_keys; ++i) keys_f16_[key_offset + i] = float_to_half(keys[i]);
        for (std::size_t i = 0; i < expected_values; ++i) values_f16_[value_offset + i] = float_to_half(values[i]);
    } else {
        for (std::size_t head = 0; head < kv_heads_; ++head) {
            for (std::size_t block = 0; block < key_blocks_per_head_; ++block) {
                const auto source_offset = head * key_dimension_ + block * 32;
                const auto encoded_offset = key_offset + source_offset;
                const auto scale_offset = slot * kv_heads_ * key_blocks_per_head_ +
                                          head * key_blocks_per_head_ + block;
                quantize_q8_block(keys + source_offset, keys_q8_.data() + encoded_offset,
                                  keys_q8_scales_.data() + scale_offset);
            }
            for (std::size_t block = 0; block < value_blocks_per_head_; ++block) {
                const auto source_offset = head * value_dimension_ + block * 32;
                const auto encoded_offset = value_offset + source_offset;
                const auto scale_offset = slot * kv_heads_ * value_blocks_per_head_ +
                                          head * value_blocks_per_head_ + block;
                quantize_q8_block(values + source_offset, values_q8_.data() + encoded_offset,
                                  values_q8_scales_.data() + scale_offset);
            }
        }
    }
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
    std::vector<float> decoded_key(precision_ == KvCachePrecision::Float16 ? key_dimension_ : 0);
    std::vector<float> decoded_value(precision_ == KvCachePrecision::Float16 ? value_dimension_ : 0);
    for (std::size_t head = 0; head < query_heads; ++head) {
        const auto kv_head = head / group_size;
        const auto* q = query + head * key_dimension_;
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t index = 0; index < size_; ++index) {
            const auto position = first_position_ + index;
            const auto slot = static_cast<std::size_t>(position % capacity_);
            const auto key_offset = slot * key_size + kv_head * key_dimension_;
            float score = 0.0F;
            if (precision_ == KvCachePrecision::Q8_0) {
                const auto scale_base = slot * kv_heads_ * key_blocks_per_head_ +
                                        kv_head * key_blocks_per_head_;
                for (std::size_t block = 0; block < key_blocks_per_head_; ++block) {
                    const auto scale = half_to_float(keys_q8_scales_[scale_base + block]);
                    score += scale * detail::dot_i8_f32(
                        q + block * 32, keys_q8_.data() + key_offset + block * 32, 32);
                }
            } else {
                const float* key = nullptr;
                if (precision_ == KvCachePrecision::Float16) {
                    for (std::size_t d = 0; d < key_dimension_; ++d) {
                        decoded_key[d] = half_to_float(keys_f16_[key_offset + d]);
                    }
                    key = decoded_key.data();
                } else key = keys_.data() + key_offset;
                score = detail::dot_f32(q, key, key_dimension_);
            }
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
            const auto value_offset = slot * value_size + kv_head * value_dimension_;
            const float probability = scores_[index] / denominator;
            if (precision_ == KvCachePrecision::Q8_0) {
                const auto scale_base = slot * kv_heads_ * value_blocks_per_head_ +
                                        kv_head * value_blocks_per_head_;
                for (std::size_t block = 0; block < value_blocks_per_head_; ++block) {
                    const auto value_scale = half_to_float(values_q8_scales_[scale_base + block]);
                    detail::scale_add_i8_f32(out + block * 32,
                        values_q8_.data() + value_offset + block * 32,
                        probability * value_scale, 32);
                }
            } else {
                const float* value = nullptr;
                if (precision_ == KvCachePrecision::Float16) {
                    for (std::size_t d = 0; d < value_dimension_; ++d) {
                        decoded_value[d] = half_to_float(values_f16_[value_offset + d]);
                    }
                    value = decoded_value.data();
                } else value = values_.data() + value_offset;
                detail::scale_add_f32(out, value, probability, value_dimension_);
            }
        }
    }
}

std::size_t KvCache::size() const noexcept { return size_; }
std::size_t KvCache::capacity() const noexcept { return capacity_; }
std::uint64_t KvCache::first_position() const noexcept { return first_position_; }
std::uint64_t KvCache::last_position() const noexcept { return last_position_; }
std::size_t KvCache::storage_bytes() const noexcept {
    return (keys_.size() + values_.size()) * sizeof(float) +
           (keys_f16_.size() + values_f16_.size() + keys_q8_scales_.size() +
            values_q8_scales_.size()) * sizeof(std::uint16_t) +
           (keys_q8_.size() + values_q8_.size()) * sizeof(std::int8_t) +
           scores_.size() * sizeof(float);
}
void KvCache::clear() noexcept {
    size_ = 0;
    has_position_ = false;
    first_position_ = 0;
    last_position_ = 0;
}

}  // namespace pokitlms
